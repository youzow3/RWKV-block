import torch, os

####################################################################################################
# Start of pytorch code
####################################################################################################

from .rwkv7_attn_pytorch import rwkv7_attn_pytorch_chunk

def load_wkv_cuda_kernel(CHUNK_LEN = 16, HEAD_SIZE = 64):
    from torch.utils.cpp_extension import load

    # load_name = f"wkv_wind_backstepping_C{HEAD_SIZE}_L{CHUNK_LEN}"
    load_name = f"wkv_wind_backstepping"

    # Check if the load_name is already loaded
    if load_name in torch.ops:
        return torch.ops[load_name]
    
    # Get the this script file path, to cmpute the cuda path
    this_file_path = os.path.dirname(os.path.abspath(__file__))

    # Load the kernel
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", "--forward-unknown-to-host-compiler"]
    load(name=load_name, sources=[f'{this_file_path}/cuda/wkv7_cuda.cu', f'{this_file_path}/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    # Return the loaded kernel
    return torch.ops[load_name]

class CudaWindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, w,q,k,v,z,b):
        CHUNK_LEN=16
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wkv_wind_backstepping.forward(state, w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y, state
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        state,w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wkv_wind_backstepping.backward(state, w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def rwkv7_attn_cuda(r,w,k,v, kk,kk_a, HEAD_SIZE=64, s0=None):
    # Preload the kernel
    load_wkv_cuda_kernel()

    # Get the shape
    B,T,HC = w.shape

    # Check if the chunk is multiple of 16
    chunk_remainder = T % 16

    # Initialize the state
    C = HEAD_SIZE
    H = HC//C

    # Initialize the state
    s0 = torch.zeros(B,H,C,C, dtype=torch.float,device=w.device) if s0 is None else s0
    sT = s0.to(dtype=torch.float)

    # Optimize the call, if chunk is multiple of 16
    if chunk_remainder == 0:
        chunk_xx, chunk_sT = rwkv7_attn_cuda_chunk(r,w,k,v, kk,kk_a, HEAD_SIZE, sT)
        return chunk_xx, chunk_sT.to(dtype=s0.dtype)

    # Compute the number of chunks
    chunks = T // 16
    si = chunks * 16

    # Get the chunked output
    chunk_xx, chunk_sT = rwkv7_attn_cuda_chunk(
        r[:,:si],w[:,:si],k[:,:si],v[:,:si], kk[:,:si],kk_a[:,:si],
        HEAD_SIZE, s0
    )

    # Get the remainder
    remain_xx, last_sT = rwkv7_attn_pytorch_chunk(
        r[:,si:],w[:,si:],k[:,si:],v[:,si:], kk[:,si:],kk_a[:,si:], 
        B, H, C, 
        torch.zeros(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
        chunk_sT, chunk_size=chunk_remainder
    )

    # Concatenate and return results
    return torch.cat([chunk_xx.to(dtype=w.dtype), remain_xx.to(dtype=w.dtype)], dim=1), last_sT.to(dtype=s0.dtype)


def rwkv7_attn_cuda_chunk(r,w,k,v, kk,kk_a, HEAD_SIZE=64, s0=None):
    '''
    Triton implementation running in blocks of 16 (hardcoded requirement for the kernel)
    '''
    B,T,HC = w.shape
    assert T % 16 == 0, 'pure cuda, only works in multiple of 16'
    C = HEAD_SIZE
    H = HC//C

    # Handling the cuda kernel
    a,b = -kk, (kk*kk_a)
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    s0 = torch.zeros(B,H,C,C, dtype=torch.float,device=w.device) if s0 is None else s0

    # Forward with backprop
    xx, st = CudaWindBackstepping.apply(s0,w,r,k,v,a,b)
    return xx.view(B,T,HC), st.view(B,H,C,C)
