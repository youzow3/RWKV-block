import torch

# Enable tensorfloat 32 
torch.set_float32_matmul_precision('high')

# Handles the RWKV v7 attention mechanic, in pure pytorch
def rwkv7_attn_pytorch(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    return rwkv7_attn_pytorch_ref(
        r,w,k,v, kk,a,
        BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
        xx, wkv_state_in
    )

####################################################################################################
# Working reference copy, that has been validated to be "identical" to the reference implementation
# However this has known pytorch compilation issues, hence the chunk wise version is used instead
@torch.compiler.disable()
def rwkv7_attn_pytorch_ref(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in
):
    ######## pure pytorch method
    # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
    ########
    vk_state = wkv_state_in.float()
    for t in range(SEQ_LEN):
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ ab.float() + vk.float())
        xx[:,t] = ((vk_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE))
    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out
####################################################################################################

def rwkv7_attn_pytorch_one_step(
    r,w,k,v, kk,a, 
    BATCH_SIZE, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in,
    offset=0
):
    '''
    Single step private varient, without the for loop
    Does not handle WKV normalization to float and dtype conversion
    '''
    # vk_state = wkv_state_in.float()
    vk_state = wkv_state_in
    
    # Single t step, at offset
    t = offset

    # Same as previous, without for loop
    r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
    vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
    ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
    vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ ab.float() + vk.float())
    xx[:,t] = (vk_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE)

    # Modified for simplicity
    # ---
    # wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    # return xx, wkv_state_out
    return xx, vk_state

def rwkv7_attn_pytorch_chunk(
    r,w,k,v, kk,a, 
    BATCH_SIZE, N_HEAD, HEAD_SIZE,
    xx, wkv_state_in,
    offset=0, chunk_size=16
):
    '''
    Chunked version of the RWKV7 attention, for better performance
    '''
    ######## pure pytorch method
    # See: https://github.com/BlinkDL/RWKV-LM/blob/d4c42b2cac10f8f3896ce153e2310dc763662b7a/RWKV-v7/rwkv_v7_demo_fast.py#L238
    ########
    vk_state = wkv_state_in.float()
    for i in range(chunk_size):
        t = offset + i
        r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
        vk = v_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ k_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        ab = (-kk_).view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1) @ (kk_*a_).view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE)
        vk_state = (vk_state * w_.view(BATCH_SIZE,N_HEAD,1,HEAD_SIZE).float() + vk_state @ ab.float() + vk.float())
        xx[:,t] = (vk_state.to(dtype=xx.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE)
    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out