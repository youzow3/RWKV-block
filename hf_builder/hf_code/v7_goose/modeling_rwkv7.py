""" RWKV Modeling"""

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ( ModelOutput,
                                            BaseModelOutputWithPast,
                                            CausalLMOutputWithPast)

import torch
from torch import nn

import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

from .configuration_rwkv7 import RWKV7Config
from .rwkv_block.v7_goose.model.rwkv7_goose_model import RWKV7GooseModel

class RWKV7PreTrainedModel(PreTrainedModel,RWKV7GooseModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = RWKV7Config
    base_model_prefix = "rwkv7"
    is_parallelizable = True
    _no_split_modules = ["RWKV7LayerBlock"]
    _keep_in_fp32_modules = []
    supports_gradient_checkpointing = True

    def _init_weights(
        self,
        module
    ):
        # Fallback to the default init weights
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            return
        elif hasattr(module, 'init_parameters'):
            module.init_parameters()
            return

        # Default FP initializer_range for Linear / LN layers
        initializer_range = 0.02

        if isinstance(module, (nn.ParameterList, nn.ModuleList)):
            # Iterate and initialize each parameter
            for param in module:
                self._init_weights(param)
        elif isinstance(module, nn.ParameterDict):
            # Iterate and initialize each parameter
            for key, param in module.items():
                self._init_weights(param)

        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

            # # RWKV does not use a blank pad idx. The pad_idx is a training token
            # if module.padding_idx is not None:
            #     module.weight.data[module.padding_idx].zero_()

@dataclass
class RWKV7Output(ModelOutput):
    """
    Class for the RWKV model outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RWKV7CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

RWKV7_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config ([`Rwkv7Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RWKV7_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary. If `past_key_values` is used, only `input_ids` that do not have their
            past calculated should be passed as `input_ids`. Indices can be obtained using [`AutoTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. [What are input
            IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        
        state (List block states, representing the RWKV various internal states per layer `(batch_size, hidden_state)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).

        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare RWKV7 Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV7_START_DOCSTRING,
)
class RWKV7Model(RWKV7PreTrainedModel):
    def __init__(self, config: RWKV7Config):
        super().__init__(config)
    
    def get_input_embeddings(self):
        return self.emb
    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            warnings.warning_once("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        
        # if output_hidden_states:
        #     warnings.warning_once("`RWKV7Model` does not `output_hidden_states` now, setting it to `False`.")
        #     output_hidden_states = False

        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        # Compute the input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids.to(self.emb.weight.device))
        x_hidden_state = inputs_embeds

        if use_cache and rwkv_state is None:
            rwkv_state = self.get_init_state(batch_size=x_hidden_state.shape[0])
        prv_stateList = rwkv_state
        ret_stateList = self.get_init_state(batch_size=x_hidden_state.shape[0], skip_init_state=True)
        
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        v_first = None
        ret_sublist = None
        
        # Lets start iterating
        for i, block in enumerate(self.blocks):
            # Build the full inner hidden state
            if output_hidden_states:
                all_hidden_states += (x_hidden_state,)

            # Forward the block
            if self.gradient_checkpointing and self.training:
                x_hidden_state, ret_sublist, v_first = self._gradient_checkpointing_func(
                    block.__call__, x_hidden_state, prv_stateList[i], v_first
                )
                ret_stateList[i] = ret_sublist
            else:
                x_hidden_state, ret_sublist, v_first = block(x_hidden_state, prv_stateList[i], v_first)
                ret_stateList[i] = ret_sublist
            
            # if output_attentions:
            #     all_attns += (ret_sublist,)
        
        # Final layer norm
        x_hidden_state = x_hidden_state.to(self.ln_out.weight.device, non_blocking=True)
        x_hidden_state = self.ln_out(x_hidden_state)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (x_hidden_state,)

        if not return_dict:
            return tuple(i for i in [x_hidden_state, rwkv_state, all_hidden_states, all_attns] if i is not None)
        return RWKV7Output(
            last_hidden_state=x_hidden_state,
            rwkv_state=rwkv_state,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )
