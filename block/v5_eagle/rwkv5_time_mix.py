import torch
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv5_block_config_map import RWKV5BlockConfigMap, RWKV5BlockConfigMapNormalizer
from .rwkv5_optimized_ops import modified_lerp, timemix_inner

class RWKV5TimeMix(torch.nn.Module):
    '''
    Time Mix block for RWKV V5
    '''

    def __init__(self, configMap: Union[RWKV5BlockConfigMap, any]):
        super().__init__()

        cMap:RWKV5BlockConfigMap = RWKV5BlockConfigMapNormalizer(configMap)
        self.configMap = cMap

        # Get required props
        n_dim = cMap.n_dim
        n_layer = cMap.n_layer

        # Get optional props
        n_dim_att = cMap.get_n_dim_att()
        layer_id = cMap.get_layer_id(0)
        device = cMap.get_device('cpu')
        dtype = cMap.get_dtype('float')

        n_head = cMap.get_n_head()
        head_size = cMap.head_size
        head_size_divisor = cMap.head_size_divisor

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = head_size_divisor

        # Build the various params
        # ---

        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_dim, device=device, dtype=dtype)
            for i in range(n_dim):
                ddd[0, 0, i] = i / n_dim

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(n_dim_att, device=device, dtype=dtype)
            for n in range(n_dim_att):
                decay_speed[n] = -6 + 5 * (n / (n_dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(n_head, head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(n_dim_att, device=device, dtype=dtype)
            for n in range(n_dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (n_dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)

        self.value = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(n_dim_att, n_dim, bias=False, device=device, dtype=dtype)
        self.gate = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.GroupNorm(n_head, n_dim_att, device=device, dtype=dtype)
        
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
        # Get the chunk length,
        # and prepare to do the inner loop
        x_chunk_len = x.size(-2)

        processed_len = 0
        remaining_len = x_chunk_len
        
        cur_shift_state = shift_state_in
        cur_wkv_state = wkv_state_in

        nxt_shift_state = None
        nxt_wkv_state = None
        nxt_output_state = None

        output_emb_arr = []

        # Chunk sizes to loop, and support
        bChunkSize_arr = [128, 16, 2, 1]

        # Iterate the bChunk
        for bChunkSize in bChunkSize_arr:
            
            # Check if the remaining length is less than the chunk size
            while remaining_len > bChunkSize:
                # Get the multiple of bChunkSize
                mul = remaining_len // bChunkSize
                chunk_len = bChunkSize * mul

                sub_chunk = x[:, processed_len:processed_len+chunk_len]

                nxt_output_state, nxt_shift_state, nxt_wkv_state = self._forward_nocuda_optimized(sub_chunk, cur_shift_state, cur_wkv_state)

                processed_len += chunk_len
                remaining_len -= chunk_len

                cur_shift_state = nxt_shift_state
                cur_wkv_state = nxt_wkv_state
                output_emb_arr.append(nxt_output_state)

        # Return the output state
        return (torch.cat(output_emb_arr, dim=-2), cur_shift_state, cur_wkv_state)
    
    # Highly optimized nocuda forward operations, bounded to specific multiples of wkv chunk lengths
    # This uses a pure pytorch implementation
    def _forward_nocuda_optimized(self, 
                                  x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor,
                                  wkv_chunk_len:int=128, wkv_precision:int=64
                                  ) -> tuple[Tensor,Tensor,Tensor]:
        shift_state_out = x[:,-1]

        # x_chunk_len = x.size(-2)
        # assert x_chunk_len % wkv_chunk_len == 0 or x_chunk_len == 1, "optimized nocuda rwkv requires data len supplied to be an exact multiple of the chunk len"

        # Get the x sizing
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size
        head_size_divisor = self.head_size_divisor
        V = K

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1)

        # Get the xk, xv, xr, xg, and rkvg
        xk = modified_lerp(x, self.time_mix_k, xx)
        xv = modified_lerp(x, self.time_mix_v, xx)
        xr = modified_lerp(x, self.time_mix_r, xx)
        xg = modified_lerp(x, self.time_mix_g, xx)

        r = self.receptance(xr).view(B, T, H, K).transpose(1, 2) # BHTK
        k = self.key(xk).view(B, T, H, K).transpose(1, 2)      # BHTK
        v = self.value(xv).view(B, T, H, V).transpose(1, 2)    # BHTV
        g = F.silu(self.gate(xg))

        w = torch.exp(-torch.exp(self.time_decay.float())).view(1,H,1,K).expand(1,H,T,K)

        u = self.time_faaaa.view(1,H,1,K).to(r.dtype)

        # Logits and state
        wkv_state_out = wkv_state_in.to(r.dtype)

        x_logits, wkv_state_out = self._timemix_inner(r, k, v, w, u, wkv_state_out, wkv_chunk_len, wkv_precision) 
        x_logits = x_logits.transpose(1,2).reshape(B,T,C)

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits / head_size_divisor).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, shift_state_out, wkv_state_out)

    @staticmethod
    def _timemix_inner(r,k,v,w,u,kv_state,chunk_len:int=128,precision:int=64)->tuple[Tensor,Tensor]:
        ''' 
        Highly optimized RWKV inner opperations, used within the TimeMix module
        This was made by @SmerkyG for supporting RWKV v5 in a pure pytorch manner
        '''

        # 24 is optimal chunk length for fp32 
        # (longer will use too much memory and cause precision problems or even numerical instability, shorter is inefficient)
        # otherwise fp64 is recommended instead
        assert(chunk_len <= 24 or precision == 64)
        """
        expects
        r : (B,H,L,K)
        k : (B,H,L,K)
        v : (B,H,L,V)
        w : (B,H,L,K) or (1,H,L,K)
        u : (1,H,1,K)
        kv_state : (B,H,K,V)
        """
        B,H,L,K = k.size()
        V = v.size(-1)
        T = chunk_len

        if L == 1:
            kv = k.mT @ v
            out = r @ (kv_state + u.mT * kv)
            kv_state = w.mT * kv_state + kv
            return out, kv_state
        else:
            # FIXME - support fast path for non-exact multiples
            # ensure it's an exact multiple
            assert L%T == 0, "fast non-cuda rwkv5.2+ requires ctxlen to be an exact multiple of chunk_len"

            N = L // T

            # this has to be done to avoid numerical instability (inf/NaN) when w is used as a divisor up to chunk_length//2 places away (so precision_min_val^(T//2) has to be in fp range)
            # NOTE - this does not account for the impact of the size of R, K so we currently use the chunk_len=32 numbers for chunk_len=24
            assert(precision == 32 or precision == 64)
            precision_min_val = 0.005 # good for fp32 (1.175e-38 ^ (1/16.0) < 0.00426)
            if precision == 32:
                precision_dtype = torch.float32
            else: #elif precision_dtype == torch.float64:
                precision_dtype = torch.float64
            w = w.clamp(precision_min_val)

            # calculate cumulative decay in log space where it won't overflow
            w_log = w.float().log() # (1,H,L,K) or (B,H,L,K)

            # chunked view of w_log
            wc_log = w_log.view(w.size(0),H,N,T,K)
            wc_log_cum = wc_log.cumsum(dim=-2)

            # chunked view of shifted_w_log
            shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

            # NOTE - we have to apply the decay weight from TWO ahead.. ONE ahead gets no decay (log==0)
            # pre-applied weights
            # left side is prior chunk (w_inter), right side is current chunk (w_intra)
            # without u...
            # w0   w1   w2   w3   | w4   w5   w6   w7          
            # w1:4 w2:4 w3:4 w4:4 | w4:5 w4:6 w4:7 w4:8
            # with u...
            # w0   w1   w2   w3   | w4   w5   w6   w7          
            # w1:4 w2:4 w3:4 w4:4 | w4:4 w4:5 w4:6 w4:7

            # ws decays the entire current state (representing t-1) to the prior block (t-2)
            ws = wc_log.sum(dim=-2, keepdim=True) # 1HN1K or BHN1K
            # w_inter is the decay to the end of the current block, since it will be applied at the next iteration when current (t) becomes prior (t-1)
            # this formula because e.g. w1:4 = w0:4 - w0:1
            w_inter = ws - wc_log_cum # 1HNTK or BHNTK (w^(T-1) ... w^0)
            # w_intra is the decay from the beginning of the current block (t), since it will be applied to current queries (t) against prior state (representing keys+values up to but not including block t)
            # this formula because e.g. w1:3 = w0:3 - w0
            w_intra = wc_log_cum - wc_log # 1HNTK or BHNTK (w^0 ... w^(T-2))

            ws = list(ws.mT.exp().to(r.dtype).unbind(dim=-3)) # N x 1HK1 or BHK1 !!NOTE THE .mT HERE!!
            w_inter = w_inter.exp().to(r.dtype) # 1HNTK or BHNTK
            w_intra = w_intra.exp().to(r.dtype) # 1HNTK or BHNTK

            # chunked view of r, k, v
            r = r.view(B,H,N,T,K) 
            k = k.view(B,H,N,T,K) 
            v = v.view(B,H,N,T,V)
            u = u.unsqueeze(2).to(r.dtype) # (1,H,1,1,K)

            # parallel calculation of all intra-chunk attention contributions
            wc_log_offset = shifted_wc_log_cum[...,T//2:T//2+1,:] # B,H,N,1,K
            r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp() # B,H,N,T,K
            k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp() # B,H,N,T,K
            a = ((r*r_decay) @ (k*k_inv_decay).mT).to(r.dtype).tril(-1) # B,H,N,T,T
            # add u term to attention (NOTE - the tril(-1) above zeroed the diagonal)
            a = a + torch.einsum('bhntk,bhntk->bhnt', r, u * k).diag_embed()
            out = a @ v # BHNTV
            # alternate way of adding in u
            # out = out + torch.einsum('bhntk,bhntk,bhntv->bhntv', r, u * k, v) 

            # parallel precalculation of chunked (k*wk).mT@v for use in recurrent state calc below
            wkv = (k * w_inter).mT @ v # BHNKV
            wkv = list(wkv.unbind(dim=-3)) # N x BHKV

            # recurrent calculation of all states
            states = []
            for i in range(N):
                states.append(kv_state)
                kv_state = kv_state * ws[i] + wkv[i] # BHKV
                # equivalent non-precalced version
                #wkv = (k[...,i,:,:] * wk[...,i,:,:]).mT @ v[...,i,:,:]
                #kv_state = kv_state * ws[i] + wkv
            states = torch.stack(states, dim=2) # BHNKV       

            # parallel application of all r to states
            out = out + (r * w_intra) @ states # BHNTV
            out = out.view(B,H,L,V)
            return out, kv_state
    
    def load_from_model_state_dict(self, state_dict: dict, layer_id:int, non_blocking:bool=True):
        '''
        Given the Full/partial RWKV model weights, loaded via `torch.load`
        Setup the RWKV_TimeMix model weights, using the layer_id
        '''
        # copy_ the values over, instead of replacing the model weights
        self.time_mix_k.data.copy_(state_dict[f"blocks.{layer_id}.att.time_mix_k"], non_blocking=non_blocking)
        self.time_mix_v.data.copy_(state_dict[f"blocks.{layer_id}.att.time_mix_v"], non_blocking=non_blocking)
        self.time_mix_r.data.copy_(state_dict[f"blocks.{layer_id}.att.time_mix_r"], non_blocking=non_blocking)
        self.time_mix_g.data.copy_(state_dict[f"blocks.{layer_id}.att.time_mix_g"], non_blocking=non_blocking)
        self.time_decay.data.copy_(state_dict[f"blocks.{layer_id}.att.time_decay"], non_blocking=non_blocking)
        self.time_faaaa.data.copy_(state_dict[f"blocks.{layer_id}.att.time_faaaa"], non_blocking=non_blocking)

        self.receptance.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.receptance.weight"], non_blocking=non_blocking)
        self.key.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.key.weight"], non_blocking=non_blocking)
        self.value.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.value.weight"], non_blocking=non_blocking)
        self.output.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.output.weight"], non_blocking=non_blocking)
        self.gate.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.gate.weight"], non_blocking=non_blocking)
        self.ln_x.weight.data.copy_(state_dict[f"blocks.{layer_id}.att.ln_x.weight"], non_blocking=non_blocking)
        self.ln_x.bias.data.copy_(state_dict[f"blocks.{layer_id}.att.ln_x.bias"], non_blocking=non_blocking)