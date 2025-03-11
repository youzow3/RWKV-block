import torch
import torch as th
import triton
import triton.language as tl

####################################################################################################
# Triton specific coding (aka mostly songlin & Johan Sokrates Wind stuff)
#
# Copyright (c) 2024, Johan Sokrates Wind, licensed under MIT
# https://github.com/johanwind/wind_rwkv/blob/main/LICENSE
####################################################################################################

# -------------------------
# Triton "smallhead" and "bighead" common code
# -------------------------

@triton.jit
def IND3(a,b,c,nb,nc):
    return (a*nb+b)*nc+c
@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def IND5(a,b,c,d,e,nb,nc,nd,ne):
    return (((a*nb+b)*nc+c)*nd+d)*ne+e

@triton.jit
def _prod(a,b): return a*b

# @triton.jit
# def _sum(a,b): return a+b

# inv(I-A) where A is a strictly lower triangular nxn matrix
@triton.jit
def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
    i = tl.arange(0,n)
    prod = (i[None,:]==i[:,None]).to(tl.float32)
    for j in range(n-1):
        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
    return prod.trans()

@triton.jit
def tl_dot(prec:tl.constexpr, a, b):
    if prec == 'fp32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
    elif prec == 'tf32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
    elif prec == 'bf16':
        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
    else:
        tl.static_assert(False)

# -------------------------
# Triton "smallhead" code
# -------------------------

@triton.jit
def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    state = tl.load(s0_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)
    for t0 in range(T//dT):
        t = t0*dT+tl.arange(0,dT)[:,None]
        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        w = (-sw.exp()).exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref
        # w = (-sw.exp())
        # fw = tl.reduce(w, 0, _sum, keep_dims=True).exp()
        # incl_pref = tl.cumsum(w,axis=0).exp()
        # non_incl_pref = incl_pref / w.exp()
        # inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2
        yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + tl_dot(prec, wq, state.trans())
        tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

        tl.store(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C), state.to(tl.float32))
        state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
    tl.store(sT_+IND4(bi,hi,i.trans(),i, H,C,C), state.to(tl.bfloat16))

@triton.jit
def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_, dw_,dq_,dk_,dv_,da_,db_,ds0_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr):
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    i = tl.arange(0,C)[None,:]
    dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),i, H,C,C)).to(tl.float32)

    for t0 in range(T//dT-1,-1,-1):
        t = t0*dT+tl.arange(0,dT)[:,None]

        state = tl.load(s_+IND5(bi,hi,t0,i.trans(),i, H,T//dT,C,C)).to(tl.float32)

        sw = tl.load(w_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sq = tl.load(q_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sk = tl.load(k_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sa = tl.load(a_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sb = tl.load(b_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
        sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

        dw_fac = -sw.exp()
        w = dw_fac.exp()
        fw = tl.reduce(w, 0, _prod, keep_dims=True)
        incl_pref = tl.cumprod(w,axis=0)
        non_incl_pref = incl_pref / w
        inv_incl_pref = 1 / incl_pref

        wq = sq * incl_pref
        wa = sa * non_incl_pref
        kwi = sk * inv_incl_pref
        bwi = sb * inv_incl_pref

        mask1 = (t > t.trans())
        ab = tl_dot(prec, wa, bwi.trans()) * mask1
        ak = tl_dot(prec, wa, kwi.trans()) * mask1

        ab_inv = tri_minv(ab, dT, prec)

        ab_u = tl_dot(prec, ak, sv) + tl_dot(prec, wa, state.trans())
        u = tl_dot(prec, ab_inv, ab_u)
        mask2 = (t >= t.trans())
        qk = tl_dot(prec, wq, kwi.trans()) * mask2
        qb = tl_dot(prec, wq, bwi.trans()) * mask2

        du = tl_dot(prec, qb.trans(), sdy) + tl_dot(prec, bwi*fw, dstate.trans())
        dab_u = tl_dot(prec, ab_inv.trans(), du)

        dv = tl_dot(prec, qk.trans(), sdy) + tl_dot(prec, kwi*fw, dstate.trans()) + tl_dot(prec, ak.trans(), dab_u)
        tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

        dab = tl_dot(prec, tl_dot(prec, ab_inv.trans(), du), u.trans()) * mask1
        dak = tl_dot(prec, dab_u, sv.trans()) * mask1
        dab_u_state = tl_dot(prec, dab_u, state)
        da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
        tl.store(da_+IND4(bi,t,hi,i, T,H,C), da.to(tl.bfloat16))

        dqb = tl_dot(prec, sdy, u.trans()) * mask2
        dqk = tl_dot(prec, sdy, sv.trans()) * mask2
        dy_state = tl_dot(prec, sdy, state)
        dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
        tl.store(dq_+IND4(bi,t,hi,i, T,H,C), dq.to(tl.bfloat16))

        fw_u_dstate = fw * tl_dot(prec, u, dstate)
        db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
        tl.store(db_+IND4(bi,t,hi,i, T,H,C), db.to(tl.bfloat16))

        fw_v_dstate = fw * tl_dot(prec, sv, dstate)
        dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
        tl.store(dk_+IND4(bi,t,hi,i, T,H,C), dk.to(tl.bfloat16))

        dw0 = fw * tl.sum(state*dstate, axis=0,keep_dims=True)
        for k in range(t0*dT,t0*dT+dT):
            lmask = (t<k).trans()
            A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
            A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
            A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
            A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
            dw = tl.sum(A, axis=0,keep_dims=True) + dw0

            wk = tl.load(w_+IND4(bi,k,hi,i, T,H,C)).to(tl.float32)
            dw *= -wk.exp()
            tl.store(dw_+IND4(bi,k,hi,i, T,H,C), dw.to(tl.bfloat16))

        dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
    tl.store(ds0_+IND4(bi,hi,i.trans(),i, H,C,C), dstate.to(tl.bfloat16))

class TritonRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        fw_attn_triton[(H,B)](w,q,k,v,z,b, s0,y,s,sT, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,z,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,z,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,dz,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,z,b,dsT]]
        bw_attn_triton[(H,B)](w,q,k,v,z,b, dy,s,dsT, dw,dq,dk,dv,dz,db,ds0, B,T,H,C,K, ctx.dot_prec)
        return dw,dq,dk,dv,dz,db,ds0,None
    
# -------------------------
# Triton "bighead" code
# -------------------------

@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def fw_attn_triton_bighead(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, wq_,wa_,kwi_,bwi_,fw_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)
    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            state = tl.load(s0_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(s_+IND5(bi,hi,0,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))

    for t0 in range(T//dT):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref
            # w = (-sw.exp())
            # fw = tl.reduce(w, 0, _sum, keep_dims=True).exp()
            # incl_pref = tl.cumsum(w,axis=0).exp()
            # non_incl_pref = incl_pref / w.exp()
            # inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            wa_state = tl.zeros((dT,dC), tl.float32)
            wq_state = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wa_state += tl_dot(prec, wa, state.trans())
                wq_state += tl_dot(prec, wq, state.trans())

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + wq_state
            tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C))

                state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)

                if t0+1 < T//dT:
                    tl.store(s_+IND5(bi,hi,t0+1,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))
                else:
                    tl.store(sT_+IND4(bi,hi,i.trans(),j, H,C,C), state.to(tl.bfloat16))


@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def bw_attn_triton_bighead(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_,ds_, dw_,dq_,dk_,dv_,da_,db_,ds0_, wq_,wa_,kwi_,bwi_,fw_,u_,dab_u_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))

    for t0 in range(T//dT-1,-1,-1):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref
            # w = (-sw.exp())
            # fw = tl.reduce(w, 0, _sum, keep_dims=True).exp()
            # incl_pref = tl.cumsum(w,axis=0).exp()
            # non_incl_pref = incl_pref / w.exp()
            # inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        dab = tl.zeros((dT,dT), tl.float32)
        dak = tl.zeros((dT,dT), tl.float32)
        dqb = tl.zeros((dT,dT), tl.float32)
        dqk = tl.zeros((dT,dT), tl.float32)

        tl.debug_barrier()
        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            wa_state = tl.zeros((dT,dC), tl.float32)
            bwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            kwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C))

                wa_state += tl_dot(prec, wa, state.trans())
                bwi_dw_dstate += tl_dot(prec, bwi*fw, dstate.trans())
                kwi_dw_dstate += tl_dot(prec, kwi*fw, dstate.trans())

            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            du = tl_dot(prec, qb.trans(), sdy) + bwi_dw_dstate
            dab_u = tl_dot(prec, ab_inv.trans(), du)

            tl.store(u_+IND4(bi,hi,dt,i, H,dT,C), u.to(tl.float32))
            tl.store(dab_u_+IND4(bi,hi,dt,i, H,dT,C), dab_u.to(tl.float32))

            dv = tl_dot(prec, qk.trans(), sdy) + kwi_dw_dstate + tl_dot(prec, ak.trans(), dab_u)
            tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

            dab += tl_dot(prec, dab_u, u.trans()) * mask1
            dak += tl_dot(prec, dab_u, sv.trans()) * mask1
            dqb += tl_dot(prec, sdy, u.trans()) * mask2
            dqk += tl_dot(prec, sdy, sv.trans()) * mask2
        tl.debug_barrier()

        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            dy_state = tl.zeros((dT,dC), tl.float32)
            dab_u_state = tl.zeros((dT,dC), tl.float32)
            fw_u_dstate = tl.zeros((dT,dC), tl.float32)
            fw_v_dstate = tl.zeros((dT,dC), tl.float32)
            state_dstate = tl.zeros((1,dC), tl.float32)

            fw = tl.load(fw_+IND3(bi,hi,j, H,C))
            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            for i0 in range(0,C,dC):
                i = i0+tl.arange(0,dC)[None,:]

                u = tl.load(u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                dab_u = tl.load(dab_u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                tl.debug_barrier()
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                tl.debug_barrier()

                dab_u_state += tl_dot(prec, dab_u, state)
                fw_u_dstate += fw * tl_dot(prec, u, dstate)
                fw_v_dstate += fw * tl_dot(prec, sv, dstate)
                dy_state += tl_dot(prec, sdy, state)

                state_dstate += tl.sum(state*dstate, axis=0,keep_dims=True)

                dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
                if t0 > 0:
                    tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))
                else:
                    tl.store(ds0_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.bfloat16))

            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref
            # w = (-sw.exp())
            # # fw = tl.reduce(w, 0, _sum, keep_dims=True).exp()
            # incl_pref = tl.cumsum(w,axis=0).exp()
            # non_incl_pref = incl_pref / w.exp()
            # inv_incl_pref = 1 / incl_pref

            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
            tl.store(da_+IND4(bi,t,hi,j, T,H,C), da.to(tl.bfloat16))

            dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
            tl.store(dq_+IND4(bi,t,hi,j, T,H,C), dq.to(tl.bfloat16))

            db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
            tl.store(db_+IND4(bi,t,hi,j, T,H,C), db.to(tl.bfloat16))

            dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
            tl.store(dk_+IND4(bi,t,hi,j, T,H,C), dk.to(tl.bfloat16))

            dw0 = fw * state_dstate
            for k in range(t0*dT,t0*dT+dT):
                lmask = (t<k).trans()
                A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
                A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
                A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
                A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
                dw = tl.sum(A, axis=0,keep_dims=True) + dw0

                wk = tl.load(w_+IND4(bi,k,hi,j, T,H,C)).to(tl.float32)
                dw *= -wk.exp()
                tl.store(dw_+IND4(bi,k,hi,j, T,H,C), dw.to(tl.bfloat16))

class TritonBigheadRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,a,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        assert T%K == 0
        assert C%16 == 0
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(4)]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        fw_attn_triton_bighead[(H,B)](w,q,k,v,a,b, s0,y,s,sT, wq,wa,kwi,bwi,fw, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        ds = th.empty(B,H,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi,u,dab_u = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(6)]
        bw_attn_triton_bighead[(H,B)](w,q,k,v,a,b, dy,s,dsT,ds, dw,dq,dk,dv,da,db,ds0, wq,wa,kwi,bwi,fw,u,dab_u, B,T,H,C,K, ctx.dot_prec)
        return dw,dq,dk,dv,da,db,ds0,None
    
####################################################################################################
# Start of pytorch code
####################################################################################################

from .rwkv7_attn_pytorch import rwkv7_attn_pytorch_chunk, rwkv7_attn_pytorch_ref_fp32

# -------------------------
# Pytorch "smallhead" code
# -------------------------

def rwkv7_attn_triton(r,w,k,v, kk,iclr, HEAD_SIZE=64, dot_prec='fp32', s0=None):
    B,T,HC = w.shape

    # Check if the chunk is multiple of 16
    chunk_remainder = T % 16

    # Initialize the state
    C = HEAD_SIZE
    H = HC//C

    # Initialize the state
    s0 = th.zeros(B,H,C,C, dtype=th.float,device=w.device) if s0 is None else s0
    sT = s0.to(dtype=torch.float).contiguous()

    # If its smaller then a chunk, use the pytorch implementation
    if T < 16:
        chunk_xx, chunk_sT = rwkv7_attn_pytorch_chunk(
            r,(-w.float().exp()).exp(),
            k,v, 
            kk,iclr, 
            B, T, H, C, 
            torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
            sT
        )
        # chunk_xx, chunk_sT = rwkv7_attn_pytorch_ref_fp32(
        #     r,w,k,v, 
        #     kk,iclr, 
        #     B, T, H, C, 
        #     torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
        #     sT
        # )
        return chunk_xx, chunk_sT.to(dtype=s0.dtype)

    # Optimize the call, if chunk is multiple of 16
    if chunk_remainder == 0:
        return rwkv7_attn_triton_chunk(r,w,k,v, kk,iclr, HEAD_SIZE, dot_prec, s0)
    
    # Compute the number of chunks
    chunks = T // 16
    si = chunks * 16

    # Get the chunked output
    chunk_xx, chunk_sT = rwkv7_attn_triton_chunk(
        r[:,:si],w[:,:si],k[:,:si],v[:,:si], kk[:,:si],iclr[:,:si],
        HEAD_SIZE, dot_prec, sT
    )

    # Get the remainder
    # ---
    remain_xx, last_sT = rwkv7_attn_pytorch_chunk(
        r[:,si:],(-w[:,si:].float().exp()).exp(),
        k[:,si:],v[:,si:], 
        kk[:,si:],iclr[:,si:], 
        B, chunk_remainder, H, C, 
        torch.empty(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
        chunk_sT
    ) 
    # remain_xx, last_sT = rwkv7_attn_pytorch_ref_fp32(
    #     r[:,si:],w[:,si:],k[:,si:],v[:,si:], 
    #     kk[:,si:],iclr[:,si:], 
    #     B, chunk_remainder, H, C, 
    #     torch.empty(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
    #     chunk_sT
    # )

    # Concatenate and return results
    return torch.cat([chunk_xx.to(dtype=w.dtype), remain_xx.to(dtype=w.dtype)], dim=1), last_sT.to(dtype=s0.dtype)


def rwkv7_attn_triton_chunk(r,w,k,v, kk,iclr, HEAD_SIZE=64, dot_prec='fp32', s0=None):
    '''
    Triton implementation running in blocks of 16 (hardcoded requirement for the kernel)
    '''
    B,T,HC = w.shape
    assert T % 16 == 0, 'pure triton, only works in multiple of 16'
    C = HEAD_SIZE
    H = HC//C

    # Moving the triton specific operations into the chunk steps
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,-kk,(kk*iclr)]]
    s0 = th.zeros(B,H,C,C, dtype=th.float,device=w.device) if s0 is None else s0
    xx, sT = TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)
    return xx.view(B,T,HC), sT

# -------------------------
# Pytorch "bighead" code
# -------------------------

def rwkv7_attn_triton_bighead(r,w,k,v, kk,iclr, HEAD_SIZE=64, dot_prec='fp32', s0=None):
    B,T,HC = w.shape

    # Check if the chunk is multiple of 16
    chunk_remainder = T % 16

    # Initialize the state
    C = HEAD_SIZE
    H = HC//C

    # Initialize the state
    s0 = th.zeros(B,H,C,C, dtype=th.float,device=w.device) if s0 is None else s0
    sT = s0.to(dtype=torch.float).contiguous()

    # If its smaller then a chunk, use the pytorch implementation
    if T < 16: 
        chunk_xx, chunk_sT = rwkv7_attn_pytorch_chunk(
            r,(-w.float().exp()).exp(),
            k,v, 
            kk,iclr, 
            B, T, H, C, 
            torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
            sT
        )
        # chunk_xx, chunk_sT = rwkv7_attn_pytorch_ref_fp32(
        #     r,w,k,v, 
        #     kk,iclr, 
        #     B, T, H, C, 
        #     torch.empty(B, T, HC, device=w.device, dtype=w.dtype),
        #     sT
        # )
        return chunk_xx, chunk_sT.to(dtype=s0.dtype)

    # Optimize the call, if chunk is multiple of 16
    if chunk_remainder == 0:
        return rwkv7_attn_triton_bighead_chunk(r,w,k,v, kk,iclr, HEAD_SIZE, dot_prec, sT)
    
    # Initialize the state
    C = HEAD_SIZE
    H = HC//C
    s0 = th.zeros(B,H,C,C, dtype=th.float,device=w.device) if s0 is None else s0

    # Compute the number of chunks
    chunks = T // 16
    si = chunks * 16

    # Get the chunked output
    chunk_xx, chunk_sT = rwkv7_attn_triton_bighead_chunk(
        r[:,:si],w[:,:si],k[:,:si],v[:,:si], kk[:,:si],iclr[:,:si],
        HEAD_SIZE, dot_prec, s0
    )

    # Get the remainder
    # ---
    remain_xx, last_sT = rwkv7_attn_pytorch_chunk(
        r[:,si:],(-w[:,si:].float().exp()).exp(),
        k[:,si:],v[:,si:], 
        kk[:,si:],iclr[:,si:], 
        B, chunk_remainder, H, C, 
        torch.zeros(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
        chunk_sT
    ) 
    # remain_xx, last_sT = rwkv7_attn_pytorch_ref_fp32(
    #     r[:,si:],w[:,si:],k[:,si:],v[:,si:], 
    #     kk[:,si:],iclr[:,si:], 
    #     B, chunk_remainder, H, C, 
    #     torch.empty(B, chunk_remainder, HC, device=w.device, dtype=w.dtype), 
    #     chunk_sT
    # )

    # Concatenate and return results
    return torch.cat([chunk_xx.to(dtype=w.dtype), remain_xx.to(dtype=w.dtype)], dim=1), last_sT.to(dtype=s0.dtype)


def rwkv7_attn_triton_bighead_chunk(r,w,k,v, kk,iclr, HEAD_SIZE=64, dot_prec='fp32', s0=None):
    '''
    Triton implementation running in blocks of 16 (hardcoded requirement for the kernel)
    '''
    B,T,HC = w.shape
    assert T % 16 == 0, 'pure triton, only works in multiple of 16'
    C = HEAD_SIZE
    H = HC//C

    # Moving the triton specific operations into the chunk steps
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,-kk,(kk*iclr)]]
    s0 = th.zeros(B,H,C,C, dtype=th.float,device=w.device) if s0 is None else s0
    xx, sT = TritonBigheadRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)
    return xx.view(B,T,HC), sT