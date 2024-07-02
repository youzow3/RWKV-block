from typing import Union
import torch, math
from torch import nn, Tensor
import torch.nn.functional as F

from ...v6_gold_finch.gold_finch_block_config_map import GoldFinchBlockConfigMap
# from .ops.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
# from .ops.norm import rms_norm

class GoldFinchGptAlpha(nn.module):
    '''
    GPT Alpha block for RWKV V6 Gold Finch model
    '''

    def __init__(self, configMap: Union[GoldFinchBlockConfigMap, any]):
        super().__init__()

        cMap:GoldFinchBlockConfigMap = GoldFinchBlockConfigMap.normalize(configMap)
        self.configMap = cMap

        # Get required props
        n_dim = cMap.n_dim
        n_layer = cMap.n_layer

        # Get optional props
        n_dim_att = cMap.get_n_dim_att()
        layer_id = cMap.get_layer_id(0)
        device = cMap.get_device('cpu')
        dtype = cMap.get_dtype('bfloat16')

        n_head = cMap.get_n_head()
        head_size = cMap.head_size
        head_size_divisor = cMap.head_size_divisor

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_dim)
            for i in range(n_dim):
                ddd[0, 0, i] = i / n_dim

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)).to(device, dtype=dtype)
            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_dim, D_MIX_DIM*3)).to(device, dtype=dtype)
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_DIM, n_dim).uniform_(-0.01, 0.01)).to(device, dtype=dtype)

        self.query = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(n_dim_att, n_dim, bias=False, device=device, dtype=dtype)
        self.ln_r = nn.LayerNorm(n_dim_att, device=device, dtype=dtype)
        self.ln_k = nn.LayerNorm(n_dim_att, device=device, dtype=dtype)
        self.ln_v = nn.LayerNorm(n_dim_att, device=device, dtype=dtype)
        self.ln_x = nn.LayerNorm(n_dim_att, device=device, dtype=dtype)


    def forward(self, x:Tensor, shift_state_in:Tensor) -> tuple[Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming states containing of shape [
            [batch_size, state_size] ## Token Shift state,
        ]
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output state of shape [
            [batch_size, state_size] ## Token Shift state,
        ]
        '''
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head

        K = IN_EMB_SIZE // N_HEAD
        V = IN_EMB_SIZE // N_HEAD

        ##########
        ## x060b2 - gptalpha
        ##########

        shift_state = x[:, -1]
        dxprev = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(BATCH_SIZE*SEQ_LEN, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)

        mr, mk, mv = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        
        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        
        q = self.ln_r(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(BATCH_SIZE,-1,N_HEAD,K).transpose(1,2)
        k = k.view(BATCH_SIZE,-1,N_HEAD,K).transpose(1,2)
        v = v.view(BATCH_SIZE,-1,N_HEAD,V).transpose(1,2)

        x = nn.functional.scaled_dot_product_attention(
            q, k, v,
            #attn_mask=self.bias_mask(lr), dropout_p=0.0, is_causal=self.bias_mask is None)
            is_causal=True)
        x = x.transpose(1,2).reshape(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
       
        x = self.ln_x(x)
        x = self.output(x)

        return x, shift_state