import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .gold_finch_block_config_map import GoldFinchBlockConfigMap
from ..v6_finch.rwkv6_fla_ops import RUN_RWKVx060_FLA

class GoldFinchTimeMix(nn.Module):
    '''
    Time Mix block for RWKV V6 Gold Finch model
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

        # # Some internal flags, to sort out
        # use_one_minus_w = True
        # use_gf_v2 = True
        # self.use_one_minus_w = use_one_minus_w
        # self.use_gf_v2 = use_gf_v2

        # Build the various params
        # ---

        # Ref:
        # https://github.com/SmerkyG/LinearAttentionArena/blob/f700ff6ae1c834cd0d4110e90d24ae90d69056c0/src/tmix_x060.py
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_dim, device=device, dtype=dtype)
            for i in range(n_dim):
                ddd[0, 0, i] = i / n_dim

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)).to(device, dtype=dtype)
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0)).to(device, dtype=dtype)
            self.time_maa_v2 = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)).to(device, dtype=dtype)

            D_MIX_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_DIM, n_dim).uniform_(-0.01, 0.01)).to(device, dtype=dtype)
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_dim, D_MIX_DIM*self.time_maa_w2.size(0))).to(device, dtype=dtype)

            # fancy time_decay
            decay_speed = torch.ones(n_dim_att)
            for n in range(n_dim_att):
                decay_speed[n] = -6 + 5 * (n / (n_dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,n_dim_att)).to(device, dtype=dtype).to(device, dtype=dtype)

            D_DECAY_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_dim, D_DECAY_DIM)).to(device, dtype=dtype)
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_DIM, n_dim_att).uniform_(-0.01, 0.01)).to(device, dtype=dtype)

            self.time_value2_w1 = nn.Parameter(torch.zeros(n_dim, D_DECAY_DIM))
            self.time_value2_w2 = nn.Parameter(torch.zeros(D_DECAY_DIM, n_dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(n_dim_att)
            for n in range(n_dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (n_dim_att - 1))) + zigzag

            ori_time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size)).to(device, dtype=dtype)
            # if self.use_gf_v2:
            self._zero_time_faaaa = torch.zeros_like(ori_time_faaaa).to(device, dtype=dtype)
            self._zero_time_faaaa.requires_grad = False
            # else:
            #   self.time_faaaa = ori_time_faaaa

        self.receptance = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)

        self.value = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(n_dim_att, n_dim, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.LayerNorm(n_dim_att, device=device, dtype=dtype)
        
    def forward(self, x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming states containing of shape [
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        ]
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output state of shape [
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        ]
        '''
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head

        ##########
        ## x060
        ##########

        shift_state_out = x[:, -1]
        dxprev = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(BATCH_SIZE*SEQ_LEN, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)

        mr, mk, mv, mw, mv2 = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        xv2 = x + dxprev * (self.time_maa_v2 + mv2)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        v2 = self.value(xv2) + torch.tanh(xv2 @ self.time_value2_w1) @ self.time_value2_w2
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2

        # if self.use_one_minus_w:
        k = k * (1 - (-w.exp()).exp())
        
        # if self.use_gf_v2:
        u = self._zero_time_faaaa
        # else:
        #     u = self.time_faaaa

        y, wkv_state_out = RUN_RWKVx060_FLA(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE, N_HEAD, r, k, v, w, u, wkv_state_in)
        
        # if self.use_gf_v2:
        y = y + v2

        y = self.ln_x(y)
        y = self.output(y)

        return y, shift_state_out, wkv_state_out

    # @torch.compile(mode="reduce-overhead", fullgraph=False)
    def forward_with_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, out_x:Tensor, shift_state_out:Tensor, wkv_state_out:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], shift_state_out[:], wkv_state_out[:] = self._forward_with_reduce_compile(in_x, shift_state_in, wkv_state_in)
        return out_x, shift_state_out, wkv_state_out

    @torch.compile(mode="reduce-overhead", fullgraph=False)
    def _forward_with_reduce_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        '''
        return self.forward(in_x, shift_state_in, wkv_state_in)
    
    # ---------------------------------
    #
    #  Model state handling
    #
    # ---------------------------------
    
    def load_from_model_state_dict(self, state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        # copy_ the values over, instead of replacing the model weights
        self.time_maa_x.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_x"], non_blocking=non_blocking)
        self.time_maa_r.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_r"], non_blocking=non_blocking)
        self.time_maa_k.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_k"], non_blocking=non_blocking)
        self.time_maa_v.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_v"], non_blocking=non_blocking)
        self.time_maa_w.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_w"], non_blocking=non_blocking)
        self.time_maa_v2.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_v2"], non_blocking=non_blocking)

        self.time_maa_w2.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_w2"], non_blocking=non_blocking)
        self.time_maa_w1.data.copy_(state_dict[f"blocks.{layer_id}.att.time_maa_w1"], non_blocking=non_blocking)

        self.time_decay.data.copy_(state_dict[f"blocks.{layer_id}.att.time_decay"], non_blocking=non_blocking)

        self.time_decay_w1.data.copy_(state_dict[f"blocks.{layer_id}.att.time_decay_w1"], non_blocking=non_blocking)
        self.time_decay_w2.data.copy_(state_dict[f"blocks.{layer_id}.att.time_decay_w2"], non_blocking=non_blocking)

        self.time_value2_w1.data.copy_(state_dict[f"blocks.{layer_id}.att.time_value2_w1"], non_blocking=non_blocking)
        self.time_value2_w2.data.copy_(state_dict[f"blocks.{layer_id}.att.time_value2_w2"], non_blocking=non_blocking)
        
        # if self.use_gf_v2 == false:
        #   self.time_faaaa.data.copy_(state_dict[f"blocks.{layer_id}.att.time_faaaa"], non_blocking=non_blocking)

        self.receptance.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.receptance.weight"], non_blocking=non_blocking)
        self.key.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.key.weight"], non_blocking=non_blocking)
        self.value.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.value.weight"], non_blocking=non_blocking)
        self.output.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.output.weight"], non_blocking=non_blocking)
        self.ln_x.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.ln_x.weight"], non_blocking=non_blocking)
        self.ln_x.bias.data.copy_(state_dict[f"blocks.{layer_id}.att.ln_x.bias"], non_blocking=non_blocking)