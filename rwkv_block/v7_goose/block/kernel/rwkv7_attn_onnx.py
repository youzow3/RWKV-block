import torch
import onnxscript
from onnxscript import opset18 as op
from onnxscript.onnx_types import FLOAT, BFLOAT16, INT64


@onnxscript.script()
def rwkv7_attn_onnx_impl(r: BFLOAT16["batch_size", "seq_len", "emb"],
                         w: BFLOAT16["batch_size", "seq_len", "emb"],
                         k: BFLOAT16["batch_size", "seq_len", "emb"],
                         v: BFLOAT16["batch_size", "seq_len", "emb"],
                         kk: BFLOAT16["batch_size", "seq_len", "emb"],
                         a: BFLOAT16["batch_size", "seq_len", "emb"],
                         BATCH_SIZE: INT64, SEQ_LEN: INT64,
                         N_HEAD: INT64, HEAD_SIZE: INT64,
                         xx: BFLOAT16["batch_size", "seq_len", "emb"],
                         wkv_state_in: FLOAT["batch_size", "n_head",
                                             "head_size", "head_size"]
                         ):
    B, T, HC = op.Shape(w)

    chunk_size = 256
    chunk_count = op.Floor(op.Div(SEQ_LEN, chunk_size))
    chunk_remainder = SEQ_LEN % chunk_size

    r_c = r[:, :chunk_size * chunk_count, :]
    w_c = w[:, :chunk_size * chunk_count, :]
    k_c = k[:, :chunk_size * chunk_count, :]
    v_c = v[:, :chunk_size * chunk_count, :]
    kk_c = kk[:, :chunk_size * chunk_count, :]
    a_c = a[:, :chunk_size * chunk_count, :]

    r_r = r[:, -chunk_remainder:, :]
    w_r = w[:, -chunk_remainder:, :]
    k_r = k[:, -chunk_remainder:, :]
    v_r = v[:, -chunk_remainder:, :]
    kk_r = kk[:, -chunk_remainder:, :]
    a_r = a[:, -chunk_remainder:, :]

    @onnxscript.graph()
    def rwkv7_attn_onnx_chunk_impl(
            wkv_state_in: FLOAT["batch_size", "n_head", "head_size", "head_size"],
            r: BFLOAT16["batch_size", "seq_len", "emb"],
            w: BFLOAT16["batch_size", "seq_len", "emb"],
            k: BFLOAT16["batch_size", "seq_len", "emb"],
            v: BFLOAT16["batch_size", "seq_len", "emb"],
            kk: BFLOAT16["batch_size", "seq_len", "emb"],
            a: BFLOAT16["batch_size", "seq_len", "emb"]
            ):
        B, NH, HS, _ = op.Shape(wkv_state_in)
        B, T, HC = op.Shape(w)

        # full_vk_ = op.MatMul(op.Reshape(v, (B, T, NH, HS, 1)),
        #                      op.Reshape(k, (B, T, NH, 1, HS)))
        full_vk_ = op.MatMul(op.Reshape(v, op.Concat(B, T, NH, HS, 1, axis=0)),
                             op.Reshape(k, op.Concat(B, T, NH, 1, HS, axis=0)))
        full_iclr_ = op.Reshape(kk * a, op.Concat(B, T, NH, 1, HS, axis=0))
        full_ab = op.MatMul(op.Reshape(-kk, op.Concat(B, T, NH, HS, 1, axis=0)), full_iclr_)

        @onnxscript.graph()
        def rwkv7_attn_onnx_wkv_impl(
                wkv_state: FLOAT["batch_size", "n_head", "head_size", "head_size"],
                w: BFLOAT16["batch_size", "emb"],
                full_vk: BFLOAT16["batch_size", "n_head", "head_size", "head_size"],
                full_ab: BFLOAT16["batch_size", "n_head", "head_size", "head_size"]):
            B, NH, HS, _ = op.Shape(wkv_state)
            B, HC = op.Shape(w)
            w_ = op.Cast(op.Reshape(w, op.Concat(B, NH, 1, HS, axis=0)), to=FLOAT.dtype)
            full_ab_ = op.Cast(full_ab, to=FLOAT.dtype)
            full_vk_ = op.Cast(full_vk, to=FLOAT.dtype)
            wkv_state = wkv_state * w_ + op.MatMul(wkv_state, full_ab_) + full_vk_
            return wkv_state, op.Cast(wkv_state, to=BFLOAT16.dtype)

        wkv_state, wkv_xx = op.Scan(wkv_state_in, full_vk_, full_ab,
                                    body=rwkv7_attn_onnx_wkv_impl,
                                    num_scan_inputs=2,
                                    scan_input_axes=[1, 1], scan_output_axes=[1])

        xx = op.MatMul(op.Cast(wkv_xx, to=BFLOAT16.dtype), op.Reshape(r, op.Concat(B, T, NH, HS, 1, axis=0)))

        return wkv_state, xx

    c_shape = op.Concat(B, -1, chunk_size, HC, axis=0)
    r_shape = op.Concat(B, 1, chunk_remainder, HC, axis=0)

    wkv_state, xx_c = op.Scan(
            wkv_state_in,
            op.Reshape(r_c, c_shape),
            op.Reshape(w_c, c_shape),
            op.Reshape(k_c, c_shape),
            op.Reshape(v_c, c_shape),
            op.Reshape(kk_c, c_shape),
            op.Reshape(a_c, c_shape),
            body=rwkv7_attn_onnx_chunk_impl, num_scan_inputs=6,
            scan_input_axes=[1] * 6, scan_output_axes=[1])

    wkv_state, xx_r = op.Scan(
            wkv_state,
            op.Reshape(r_r, r_shape),
            op.Reshape(w_r, r_shape),
            op.Reshape(k_r, r_shape),
            op.Reshape(v_r, r_shape),
            op.Reshape(kk_r, r_shape),
            op.Reshape(a_r, r_shape),
            body=rwkv7_attn_onnx_chunk_impl, num_scan_inputs=6,
            scan_input_axes=[1] * 6, scan_output_axes=[1])

    xx_shape = op.Concat(B, -1, HC, axis=0)
    xx = op.Concat(op.Reshape(xx_c, xx_shape),
                   op.Reshape(xx_r, xx_shape), axis=1)

    return xx, wkv_state


def rwkv7_attn_onnx(r, w, k, v, kk, a,
                    BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
                    xx, wkv_state_in):
    xx, wkv_state = torch.ops.rwkv.rwkv7_attn(
            r, w, k, v, kk, a,
            BATCH_SIZE, SEQ_LEN, N_HEAD, HEAD_SIZE,
            xx, wkv_state_in)
    return xx, wkv_state


registory: torch.onnx.OnnxRegistry = torch.onnx.OnnxRegistry()
registory.register_op(
        rwkv7_attn_onnx_impl.to_function_proto(), "rwkv", "rwkv7_attn")
assert registory.is_registered_op("rwkv", "rwkv7_attn")
