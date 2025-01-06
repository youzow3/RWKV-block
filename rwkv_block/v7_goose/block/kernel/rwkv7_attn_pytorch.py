import torch

# We intentionally disable the compiler, as the pure pytorch implementation
# is known to be "unstable" for pytorch compile
@torch.compiler.disable()
def rwkv7_attn_pytorch(
    r,w,k,v, kk,a, 
    BATCH_SIZE, SEQ_LEN, IN_EMB_SIZE, N_HEAD, HEAD_SIZE,
    x, xx, wkv_state_in
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
        xx[:,t] = (vk_state.to(dtype=x.dtype) @ r_.view(BATCH_SIZE,N_HEAD,HEAD_SIZE,1)).view(BATCH_SIZE,N_HEAD*HEAD_SIZE)
    wkv_state_out = vk_state.to(dtype=wkv_state_in.dtype)
    return xx, wkv_state_out