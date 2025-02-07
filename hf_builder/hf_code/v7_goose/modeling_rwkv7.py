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
from transformers.modeling_outputs import ModelOutput

import torch, math
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union, Any

# Load the RWKV7Config and RWKV7GooseModel
from .configuration_rwkv7 import RWKV7Config
from .modeling_blocks_rwkv7 import RWKV7GooseModel

class RWKV7PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """
    config_class = RWKV7Config
    
    base_model_prefix = "model"
    is_parallelizable = True
    _no_split_modules = ["RWKV7LayerBlock"]
    _keep_in_fp32_modules = []

    # Enable gradient checkpointing by default
    supports_gradient_checkpointing = True
    gradient_checkpointing = True

    def __init__(self, config: RWKV7Config=None):
        if config is None and self.config is not None:
            config = self.config
        else:
            self.config = config
        if config is None:
            raise ValueError("Missing `config` or `config` attribute in the class")
            
        super().__init__(config)
        
    def _init_weights(
        self,
        module
    ):
        # Fallback to the default init weights
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
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
    "The bare RWKV7 Model transformer outputting raw hidden-states without activating the head (variable is still declared)",
    RWKV7_START_DOCSTRING,
)
class RWKV7Model(RWKV7GooseModel, RWKV7PreTrainedModel):
    def __init__(self, config: RWKV7Config):
        # Work around for multiple inheritance
        self.config = config
        super().__init__(config)
        # RWKV7GooseModel.__init__(self,config)
        # RWKV7PreTrainedModel.__init__(self,config)
    
    def get_input_embeddings(self):
        return self.emb
    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.lm_head
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV7_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        output_type=RWKV7Output,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # not in use
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, RWKV7Output]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_attentions:
            warnings.warning_once("`RWKV7Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False
 
        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if output_hidden_states:
            warnings.warning_once("`RWKV7Model` does not `output_hidden_states` now, setting it to `False`.")
            output_hidden_states = False

        # ---

        # Compute the input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids.to(self.emb.weight.device))
        x_hidden_state = inputs_embeds

        # Input length to perform chunking by
        batch_size = x_hidden_state.shape[0]
        x_input_length = x_hidden_state.shape[1]

        # Initialize the rwkv_state / prv_stateList
        if rwkv_state is None or use_cache == False:
            rwkv_state = self.get_init_state(batch_size=batch_size)
        prv_stateList = rwkv_state

        # Initialize the ret_stateList
        ret_stateList = self.get_init_state(batch_size=batch_size, skip_init_state=True)
       
        # Internal states
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        v_first = None
        ret_sublist = None

        # Get the forward chunk size, and the chunk count
        forward_chunk_size = self.config.forward_chunk_size
        forward_chunk_count = math.ceil( x_input_length / forward_chunk_size )

        # Block forward, with gradient if needed
        def block_forward(block, in_x_state, in_rwkv_state, in_v_first):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(
                    block.__call__, in_x_state, in_rwkv_state, in_v_first
                )
            else:
                return block(in_x_state, in_rwkv_state, in_v_first)
        
        # Lets start iterating the blocks
        for i, block in enumerate(self.blocks):
            # Build the full inner hidden state
            if output_hidden_states:
                all_hidden_states += (x_hidden_state,)

            # Forward the block as it is
            if forward_chunk_count <= 1:
                x_hidden_state, ret_sublist, v_first = block_forward(block, x_hidden_state, prv_stateList[i], v_first)
                ret_stateList[i] = ret_sublist
            else:
                # Damn it, we need to chunk
                new_x_hidden_state_arr = [None]*forward_chunk_count
                v_first_arr = [None]*forward_chunk_count if v_first is None else None
                ret_subList = prv_stateList[i]

                # Forward in chunks
                for chunk_idx in range(forward_chunk_count):
                    start = chunk_idx * forward_chunk_size
                    endin = min(start + forward_chunk_size, x_input_length)

                    new_x_hidden_state, ret_subList, v_first_part = block_forward(
                        block, 
                        x_hidden_state[:, start:endin], 
                        ret_subList, 
                        v_first[:, start:endin] if v_first is not None else None
                    )

                    new_x_hidden_state_arr[chunk_idx] = new_x_hidden_state
                    if v_first_arr is not None:
                        v_first_arr[chunk_idx] = v_first_part

                # Merge the forward chunks, save the state
                x_hidden_state = torch.cat(new_x_hidden_state_arr, dim=1)
                if v_first_arr is not None:
                    v_first = torch.cat(v_first_arr, dim=1)
                ret_stateList[i] = ret_subList

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

# @add_start_docstrings(
#     """
#     The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input
#     embeddings).
#     """,
#     RWKV7_START_DOCSTRING,
# )
class RWKV7ForCausalLM(RWKV7Model, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def prepare_inputs_for_generation(
        self, 
        input_ids=None, 
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None,
        # num_new_tokens_if_rwkv_state: int = 1, # Only triggers if given input_ids + rwkv_state
        num_logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        '''
        Personal Notes: On huggingface barely documented "Transformer" hooks.

        I assume this is triggered once, for the start of AI inference.
        With subsequent calls for forward on each token step, being updated with
        `_update_model_kwargs_for_generation` function instead?
        '''
        # # only last token for `inputs_ids` if the `past_key_values` is passed along.
        # if rwkv_state is not None and input_ids is not None:
        #     input_ids = input_ids[:, -num_new_tokens_if_rwkv_state:]
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError("You cannot specify both `inputs_ids` and `inputs_embeds` at the same time")
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep

        model_inputs.update({
            'rwkv_state': rwkv_state,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'num_logits_to_keep': num_logits_to_keep,
        })
        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        # Overwritten -- this model uses `state`, but doesn't have a cache (`past_key_values`)
        rwkv_state = outputs.get("rwkv_state", None)
        input_ids = model_kwargs.get("input_ids", None)
        attention_mask = model_kwargs.get("attention_mask", None)

        # only last token for inputs_ids if the state is passed along.
        if rwkv_state is not None and input_ids is not None and num_new_tokens > 0:
            input_ids = input_ids[:, -num_new_tokens:]
            model_kwargs["input_ids"] = input_ids

            if attention_mask is not None:
                attention_mask = attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))
                model_kwargs["attention_mask"] = attention_mask

        # Return the formated output
        return model_kwargs
    
    # @add_start_docstrings_to_model_forward(RWKV7_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     output_type=RWKV7CausalLMOutput,
    # )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        rwkv_state: Optional[list[tuple[torch.Tensor,torch.Tensor,torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, RWKV7CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rwkv_outputs = RWKV7Model.forward(
            self, input_ids, attention_mask, inputs_embeds, 
            rwkv_state, use_cache, output_attentions, output_hidden_states,
            return_dict=False
        )

        # Get the hidden state, and the updated RWKV state
        hidden_states = rwkv_outputs[0]
        rwkv_state = rwkv_outputs[1]

        # Get the ALL hidden states and attentions dumps
        all_hidden_states = rwkv_outputs[2] if output_hidden_states else None
        if output_hidden_states:
            all_attns = rwkv_outputs[3] if output_attentions else None
        else:
            all_attns = rwkv_outputs[2] if output_attentions else None
        
        # Forward the head state
        logits = self.head(hidden_states)

        # Compute the loss from the labels
        loss = None
        if labels is not None:

            # Setup loss function
            if self._loss_function_cache is None:
                self._loss_function_cache = CrossEntropyLoss()

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Compute the token loss

            if attention_mask is not None:
                token_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                submask = attention_mask[..., 1:].contiguous().view(-1)
                loss = (token_loss * submask).sum() / submask.sum()
            else:
                loss = F.cross_entropy(shift_logits.view(-1, shift_labels.size(-1)), shift_labels.view(-1), reduction="mean")

        if not return_dict:
            return tuple(i for i in [loss, logits, rwkv_state, all_hidden_states, all_attns] if i is not None)
        
        return RWKV7CausalLMOutput(
            loss=loss,
            logits=logits,
            rwkv_state=rwkv_state,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )

__all__ = ["RWKV7ForCausalLM", "RWKV7Model", "RWKV7PreTrainedModel"]