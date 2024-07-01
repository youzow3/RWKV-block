from typing import Union
import torch, math
from torch import nn, Tensor
import torch.nn.functional as F

from .gold_finch_block_config_map import GoldFinchBlockConfigMap
from .ops.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from .ops.norm import rms_norm

class GPTAlpha_Tmix(nn.module):
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

    # def __init__(self, args, layer_id, angles, bias_mask):
    #     super().__init__()
    #     self.args = args
    #     self.layer_id = layer_id
    #     self.n_layer = args.n_layer

    #     self.k_head_size = self.v_head_size = self.head_size = args.head_size_a
    #     self.n_kv_head = self.n_head = args.dim_att // self.head_size
    #     assert args.dim_att % self.n_head == 0

    #     with torch.no_grad():
    #         ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
    #         ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
    #         ddd = torch.ones(1, 1, args.n_embd)
    #         for i in range(args.n_embd):
    #             ddd[0, 0, i] = i / args.n_embd

    #         self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
    #         self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
    #         self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
    #         self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
    #         D_MIX_LORA = 32
    #         self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
    #         self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

    #     self.query = nn.Linear(args.n_embd, args.dim_att, bias=False)
    #     self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
    #     self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
    #     self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
    #     self.ln_r = nn.LayerNorm(args.dim_att)
    #     self.ln_k = nn.LayerNorm(args.dim_att)
    #     self.ln_v = nn.LayerNorm(args.dim_att)
    #     self.ln_x = nn.LayerNorm(args.dim_att)

    #     self.angles = angles
    #     self.bias_mask = bias_mask

    # @MyFunction
    # def forward(self, x, xo, kv_cache, last_time_mix_state:TimeMixState):
    #     B, T, C = x.size()
    #     H = self.n_head
    #     K = C // H
    #     V = C // H

    #     shift_state = x[:, -1].clone()
    #     dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

    #     xxx = x + dxprev * self.time_maa_x

    #     xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
    #     xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

    #     mr, mk, mv = xxx.unbind(dim=0)
    #     xq = x + dxprev * (self.time_maa_r + mr)
    #     xk = x + dxprev * (self.time_maa_k + mk)
    #     xv = x + dxprev * (self.time_maa_v + mv)
        
    #     q = self.query(xq)
    #     k = self.key(xk)
    #     v = self.value(xv)
        
    #     q = self.ln_r(q)
    #     k = self.ln_k(k)
    #     v = self.ln_v(v)

    #     q = q.view(B,-1,H,K).transpose(1,2)
    #     k = k.view(B,-1,H,K).transpose(1,2)
    #     v = v.view(B,-1,H,V).transpose(1,2)

    #     if self.angles is not None:
    #         self.angles = self.angles.to(x.device)
    #         q, k = apply_rotary_embedding(q, k, self.angles)

    #     x = nn.functional.scaled_dot_product_attention(
    #         q, k, v,
    #         #attn_mask=self.bias_mask(lr), dropout_p=0.0, is_causal=self.bias_mask is None)
    #         is_causal=True)
    #     x = x.transpose(1,2).reshape(B,T,C)
       
    #     x = self.ln_x(x)

    #     x = self.output(x)

    #     return x, TimeMixState(last_time_mix_state.wkv_state, shift_state)