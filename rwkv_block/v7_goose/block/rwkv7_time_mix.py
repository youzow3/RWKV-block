import torch, math
from torch import nn
from torch import Tensor
from typing import Union
from torch.nn import functional as F

from .rwkv7_block_config_map import RWKV7BlockConfigMap

# Check for triton package, if GPU is available
triton = None
if torch.cuda.is_available():
    try:
        import triton
    except ImportError:
        triton = None
    if triton is not None:
        from .kernel.rwkv7_attn_triton import rwkv7_attn_triton
else:
    print("[WARNING] Triton not available, falling back to pytorch mode by default - this is significantly slower")

# Pure pytorch mode for rwkv attention
from .kernel.rwkv7_attn_pytorch import rwkv7_attn_pytorch

class RWKV7TimeMix(torch.nn.Module):
    '''
    Time Mix block for RWKV V7
    '''

    def __init__(self, configMap: Union[RWKV7BlockConfigMap, any]):
        '''
        Initialize the TimeMix block.
        
        Note: this does not initialize the parameter weights itself
        which would depend on the `init_parameters()` method
        '''
        super().__init__()

        configMap:RWKV7BlockConfigMap = RWKV7BlockConfigMap.normalize(configMap)
        self.configMap = configMap

        # Get required props
        n_dim = configMap.n_dim
        n_layer = configMap.n_layer

        # Get the layer id
        layer_id = configMap.get_layer_id(0)
        self.layer_id = layer_id

        # Get optional props
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')

        # By default, n_dim_ffn = n_dim
        n_dim_att = configMap.get_n_dim_att()

        # Assert n_dim == n_dim_att, until we support different n_dim and n_dim_att
        assert n_dim == n_dim_att, "n_dim should be equal to n_dim_att (@TODO: support different n_dim and n_dim_att)"

        # Head size settings
        head_size = configMap.head_size
        self.head_size = head_size
        head_size_divisor = configMap.head_size_divisor

        # Number of heads
        n_head = n_dim_att // head_size
        assert n_dim_att % head_size == 0, "n_dim_att should be divisible by head_size"
        self.n_head = n_head

        # Backend
        self.tmix_backend = configMap.tmix_backend

        # Build the various params
        # ---

        with torch.no_grad():
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            def calc_lora_rank(exponent, multiplier):
                return max(1, round(n_dim ** exponent * multiplier / 32)) * 32
            D_DECAY_LORA = calc_lora_rank(0.5, 1.8)
            D_AAA_LORA   = calc_lora_rank(0.5, 1.8)
            D_MV_LORA    = calc_lora_rank(0.5, 1.3)
            D_GATE_LORA  = calc_lora_rank(0.8, 0.6)

            self.x_r = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.x_w = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.x_k = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.x_v = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.x_a = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.x_g = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))

            self.w0 = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.w1 = nn.Parameter(torch.empty(n_dim, D_DECAY_LORA, device=device, dtype=dtype))
            self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, n_dim, device=device, dtype=dtype))

            self.a0 = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.a1 = nn.Parameter(torch.empty(n_dim,D_AAA_LORA, device=device, dtype=dtype))
            self.a2 = nn.Parameter(torch.empty(D_AAA_LORA,n_dim, device=device, dtype=dtype))
            
            if layer_id > 0:
                self.v0 = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
                self.v1 = nn.Parameter(torch.empty(n_dim,D_MV_LORA, device=device, dtype=dtype))
                self.v2 = nn.Parameter(torch.empty(D_MV_LORA,n_dim, device=device, dtype=dtype))
                
            self.g1 = nn.Parameter(torch.empty(n_dim, D_GATE_LORA, device=device, dtype=dtype))
            self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, n_dim, device=device, dtype=dtype))

            self.k_k = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.k_a = nn.Parameter(torch.empty(1,1,n_dim, device=device, dtype=dtype))
            self.r_k = nn.Parameter(torch.empty(n_head, head_size, device=device, dtype=dtype))

        self.receptance = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(n_dim_att, n_dim, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.GroupNorm(n_head, n_dim_att, device=device, dtype=dtype, eps=(1e-5)*head_size)
        
    def init_parameters(self):
        '''
        Reset the parameters of the block, to an initial state used for training a model from scratch
        '''
        configMap = self.configMap

        # Get required props
        n_dim = configMap.n_dim
        n_layer = configMap.n_layer

        # Get the layer id
        layer_id = self.layer_id

        # Get optional props
        device = configMap.get_device('cpu')
        dtype = configMap.get_dtype('bfloat16')

        # By default, n_dim_ffn = n_dim
        n_dim_att = configMap.get_n_dim_att()

        # Assert n_dim == n_dim_att, until we support different n_dim and n_dim_att
        assert n_dim == n_dim_att, "n_dim should be equal to n_dim_att (@TODO: support different n_dim and n_dim_att)"

        # Head size settings
        head_size = self.head_size
        head_size_divisor = configMap.head_size_divisor

        # Number of heads
        n_head = n_dim_att // head_size
        assert n_dim_att % head_size == 0, "n_dim_att should be divisible by head_size"
        
        # Reset the various params
        # ---
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_dim, device=device, dtype=dtype)
            for i in range(n_dim):
                ddd[0, 0, i] = i / n_dim

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            def calc_lora_rank(exponent, multiplier):
                return max(1, round(n_dim ** exponent * multiplier / 32)) * 32
            D_DECAY_LORA = calc_lora_rank(0.5, 1.8)
            D_AAA_LORA   = calc_lora_rank(0.5, 1.8)
            D_MV_LORA    = calc_lora_rank(0.5, 1.3)
            D_GATE_LORA  = calc_lora_rank(0.8, 0.6)

            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v.copy_(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                x = x.to(device)
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                else:
                    assert False
                return x.to(device, dtype=dtype)

            # D_DECAY_LORA = max(32, int(round(  (1.8*(n_dim**0.5))  /32)*32))
            decay_speed = torch.ones(n_dim, device=device, dtype=dtype)
            for n in range(n_dim):
                decay_speed[n] = -7 + 5 * (n / (n_dim - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            
            self.w0.copy_(decay_speed.reshape(1,1,n_dim).to(device, dtype=dtype) + 0.5)  # !!! 0.5 comes from F.softplus !!!
            self.w1.copy_(torch.zeros(n_dim, D_DECAY_LORA, device=device, dtype=dtype))
            self.w2.copy_(ortho_init(torch.zeros(D_DECAY_LORA, n_dim), 0.1))

            # D_AAA_LORA = max(32, int(round(  (1.8*(n_dim**0.5))  /32)*32)) # suggestion
            self.a0.copy_(torch.zeros(1,1,n_dim, device=device, dtype=dtype))
            self.a1.copy_(torch.zeros(n_dim, D_AAA_LORA, device=device, dtype=dtype))
            self.a2.copy_(ortho_init(torch.zeros(D_AAA_LORA, n_dim), 0.1))

            # D_MV_LORA = max(32, int(round(  (1.3*(n_dim**0.5))  /32)*32)) # suggestion
            if layer_id > 0:
                self.v0.copy_(torch.zeros(1,1,n_dim, device=device, dtype=dtype)+1.0)
                self.v1.copy_(torch.zeros(n_dim, D_MV_LORA, device=device, dtype=dtype))
                self.v2.copy_(ortho_init(torch.zeros(D_MV_LORA, n_dim), 0.1))

            # D_GATE_LORA = max(32, int(round(  (0.6*(n_dim**0.8))  /32)*32)) # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1.copy_(torch.zeros(n_dim, D_GATE_LORA, device=device, dtype=dtype))
            self.g2.copy_(ortho_init(torch.zeros(D_GATE_LORA, n_dim), 0.1))

            self.k_k.copy_(torch.ones(1,1,n_dim, device=device, dtype=dtype)*0.85)
            self.k_a.copy_(torch.ones(1,1,n_dim, device=device, dtype=dtype))
            self.r_k.copy_(torch.zeros(n_head,head_size, device=device, dtype=dtype))
            
        self.receptance = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.key = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.value = nn.Linear(n_dim, n_dim_att, bias=False, device=device, dtype=dtype)
        self.output = nn.Linear(n_dim_att, n_dim, bias=False, device=device, dtype=dtype)
        self.ln_x = nn.GroupNorm(n_head, n_dim_att, device=device, dtype=dtype, eps=(1e-5)*head_size)

    def forward(self, x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        forwarding time mix given the model weights and the input tokens and states.
        
        Given:
        - Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
        - Incoming states containing of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        - Incoming v_first_val of shape [batch_size, seq_len, embedding_size]
        
        
        Returns a pair 
        - output embedding of shape [batch_size, seq_len, embedding_size]
        - output state of shapes:
            [batch_size, state_size] ## Token Shift state,
            [batch_size, n_head, head_size, head_size] ## WKV state
        - output v_first_val of shape [batch_size, seq_len, embedding_size]
        
        '''
        # Get the sizing
        BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE = x.size()
        N_HEAD = self.n_head
        HEAD_SIZE = self.head_size

        ##########
        ## x070
        ##########

        shift_state_out = x[:, -1]
        dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xr = x + dxprev * self.x_r
        xw = x + dxprev * self.x_w
        xk = x + dxprev * self.x_k
        xv = x + dxprev * self.x_v
        xa = x + dxprev * self.x_a
        xg = x + dxprev * self.x_g
        xx = dxprev

        r = self.receptance(xr)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1), dim=-1, p=2.0).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        k = k * (1 + (a-1) * self.k_a)

        if self.layer_id == 0:
            v_first_val = v # store the v of the first layer
        else:
            v = v + (v_first_val - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual

        tmix_backend = self.tmix_backend
        if tmix_backend == "auto":
            if triton is None or self.receptance.weight.device.type == "cpu":
                tmix_backend = "pytorch"
            else:
                tmix_backend = "triton"

        if tmix_backend == "pytorch":
            ######## pure pytorch method
            # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
            ########
            w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + w).float())) # 0.606531 = exp(-0.5)
            xx, wkv_state_out = rwkv7_attn_pytorch(r, w, k, v, kk, a, BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE, xx, wkv_state_in) 
        elif tmix_backend == "triton":
            w = -F.softplus(-(self.w0 + w)) - 0.5
            xx, wkv_state_out = rwkv7_attn_triton(r, w, k, v, kk, a, s0=wkv_state_in.clone())
        else:
            raise ValueError(f"Unknown tmix_backend: {tmix_backend}")

        # wkv_state_in normalization of type
        if wkv_state_in is not None:
            wkv_state_out = wkv_state_out.to(wkv_state_in.dtype)

        ######## cuda-based method 
        # wkv_state_out = wkv_state_in.clone()
        # w = -F.softplus(-(self.w0 + w)) - 0.5 # soft-clamp to (-inf, -0.5)
        # xx = RWKV7_OP(wkv_state_out, r, w, k, v, -kk, kk*a)
        ######## cuda-based method 

        xx = self.ln_x(xx.view(BATCH_SIZE * SEQ_LEN, IN_EMB_SIZE)).view(BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE)
        xx = xx + ((r.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*k.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(BATCH_SIZE,SEQ_LEN,N_HEAD,-1)).view(BATCH_SIZE,SEQ_LEN,IN_EMB_SIZE)
        xx = self.output(xx * g)

        return xx, shift_state_out, wkv_state_out, v_first_val

    @torch.compile(mode="default")
    def forward_with_default_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val_in:Tensor, out_x:Tensor, shift_state_out:Tensor, wkv_state_out:Tensor, v_first_val_out:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no new tensors being created for the output
        Useful for static memory allocation optimizations inference
        '''
        out_x[:], shift_state_out[:], wkv_state_out[:], v_first_val_out[:] = self.forward(in_x, shift_state_in, wkv_state_in, v_first_val_in)
        return out_x, shift_state_out, wkv_state_out, v_first_val_out

    @torch.compile(mode="reduce-overhead")
    def forward_with_reduce_compile(self, in_x:Tensor, shift_state_in:Tensor, wkv_state_in:Tensor, v_first_val:Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        '''
        Compiled varient of the forward function
        With no input tensor being modified. 
        Useful for reduce-overhead compile mode
        '''
        return self.forward(in_x, shift_state_in, wkv_state_in, v_first_val)
    
    # ---------------------------------
    #
    #  Model state handling
    #
    # ---------------------------------
    
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
            try:
                current_state_dict[n].copy_(model_state_dict[model_key], non_blocking=non_blocking)
            except Exception as e:
                print(f"[ERROR] loading: {model_key} | model shape: {current_state_dict[n].shape} | weight shape: {model_state_dict[model_key].shape}")
                raise e