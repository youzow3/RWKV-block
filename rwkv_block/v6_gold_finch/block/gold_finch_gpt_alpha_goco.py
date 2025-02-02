from typing import Union
import torch, math
from torch import nn, Tensor
import torch.nn.functional as F

from .gold_finch_block_config_map import GoldFinchBlockConfigMap
from .ops.rotary import apply_rotary_embedding # generate_rotary_embedding, generate_binary_rotary_embedding, 
from .ops.norm import rms_norm 

class GoldFinchGPTAlphaGoCo(nn.Module):
    '''
    GPT Alpha block for RWKV V6 Gold Finch model GOCO varient
    '''

    def __init__(self, configMap: Union[GoldFinchBlockConfigMap, any]):
        super().__init__()

        configMap:GoldFinchBlockConfigMap = GoldFinchBlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        hidden_size = configMap.hidden_size
        num_hidden_layers = configMap.num_hidden_layers

        # Get optional props
        hidden_size_att = configMap.get_hidden_size_att()
        layer_id = configMap.get_layer_id(0)
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')

        n_head = configMap.get_n_head()
        head_size = configMap.head_size
        head_size_divisor = configMap.head_size_divisor

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size
            
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_q = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)).to(device, dtype=dtype)

            self.time_maa_v_cache = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)).to(device, dtype=dtype)
            
            D_MIX_DIM = 32
            self.time_maa_q_w1 = nn.Parameter(torch.zeros(hidden_size, D_MIX_DIM)).to(device, dtype=dtype)
            self.time_maa_q_w2 = nn.Parameter(torch.empty(D_MIX_DIM, hidden_size).uniform_(-0.01, 0.01)).to(device, dtype=dtype)
            self.time_maa_kv_w1 = nn.Parameter(torch.zeros(hidden_size, D_MIX_DIM*2)).to(device, dtype=dtype)
            self.time_maa_kv_w2 = nn.Parameter(torch.empty(2, D_MIX_DIM, hidden_size).uniform_(-0.01, 0.01)).to(device, dtype=dtype)

            D_VALUE_DIM = max(hidden_size // 16, 64)
            self.time_key_w1 = nn.Parameter(torch.zeros(hidden_size, D_VALUE_DIM)).to(device, dtype=dtype)
            self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_DIM, hidden_size_att).uniform_(-0.01, 0.01)).to(device, dtype=dtype)
            self.time_value_w1 = nn.Parameter(torch.zeros(hidden_size, D_VALUE_DIM)).to(device, dtype=dtype)
            self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_DIM, hidden_size_att).uniform_(-0.01, 0.01)).to(device, dtype=dtype)

        self.query = nn.Linear(hidden_size, hidden_size_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, device=device, dtype=dtype)
        self.ln_q = nn.LayerNorm(hidden_size_att, device=device, dtype=dtype)
        self.ln_k = nn.LayerNorm(hidden_size_att, device=device, dtype=dtype)
        self.ln_v = nn.LayerNorm(hidden_size_att, device=device, dtype=dtype)
        self.ln_x = nn.LayerNorm(hidden_size_att, device=device, dtype=dtype)

        # timeshifting util
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x:Tensor, shift_state_in:Tensor, xo:Tensor, k_cache:Tensor) -> tuple[Tensor,Tensor]:
        '''
        forwarding gptalpha given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming shift state of shape [batch_size, state_size]
        - x_original_cache
        - kv_cache

        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output shift state of shape [batch_size, state_size]
        '''
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head

        K = IN_EMB_SIZE // N_HEAD
        V = IN_EMB_SIZE // N_HEAD

        assert xo is not None
        assert k_cache is not None

        ##########
        ## x060b2 - gptalpha goco
        ##########

        shift_state = x[:, -1]
        dxprev = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        mq = torch.tanh(xxx @ self.time_maa_q_w1) @ self.time_maa_q_w2

        xo = rms_norm(xo)
        dxo_prev = self.time_shift(xo) - xo
        xxx = xo + dxo_prev * self.time_maa_v_cache
        xxx = torch.tanh(xxx @ self.time_maa_kv_w1).view(B*xo.size(1), self.time_maa_kv_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_kv_w2).view(self.time_maa_kv_w2.size(0), BATCH_SIZE, xo.size(1), IN_EMB_SIZE)
        mk, mv = xxx.unbind(dim=0)

        k = k_cache
        dkprev = self.time_shift(k) - k
        v = xo
        dvprev = self.time_shift(v) - v

        xq = x + dxprev * (self.time_maa_q + mq)
        k = k + dkprev * (self.time_maa_k + mk)
        v = v + dvprev * (self.time_maa_v + mv)

        k = k + torch.tanh(k @ self.time_key_w1) @ self.time_key_w2
        v = v + torch.tanh(v @ self.time_value_w1) @ self.time_value_w2     

        q = self.query(xq)
        
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(BATCH_SIZE,-1,N_HEAD,K).transpose(1,2)
        k = k.view(BATCH_SIZE,-1,N_HEAD,K).transpose(1,2)
        v = v.view(BATCH_SIZE,-1,N_HEAD,V).transpose(1,2)

        if self.angles is not None:
            self.angles = self.angles.to(x.device)
            q, k = apply_rotary_embedding(q, k, self.angles)

        # causality MUST be enforced for longer runs because even though we won't use the results at t-1 the next chanmix WILL for its tokenshift!
        # this is also why we must allow through the last MANY time-steps if we have that many, so chanmix receives both of these and can lerp between those results!
        # the results can tokenshift their way forward up to one full timestep each layer via chanmix, so we really have to keep up to all N goldfinch layers around

        x = nn.functional.scaled_dot_product_attention(q,k,v,is_causal=q.size(-2)>1)

        x = x.transpose(1,2).reshape(BATCH_SIZE,-1,IN_EMB_SIZE)
       
        x = self.ln_x(x)
        #x = F.layer_norm(x.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(x.dtype)

        x = self.output(x)

        return x, shift_state
    
    def load_from_model_state_dict(self, model_state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the the current module weights, using the layer_id
        '''
        # Get the current state_dict
        current_state_dict = self.state_dict()

        # Iterate each parameter in the state_dict, and compare from the model
        for n in current_state_dict:
            model_key = f"blocks.{layer_id}.att.{n}"
            if model_key not in model_state_dict:
                continue

            # Copy the values from the state_dict
            current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)