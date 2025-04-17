import torch

import rwkv7_attn_pytorch
import rwkv7_attn_onnx

import unittest


class Test(unittest.TestCase):
    def test_ref(self):
        B, T, HC = 32, 64, 128
        NH, HS = 4, 32
        r = torch.randn((B, T, HC), dtype=torch.bfloat16)
        w = torch.randn((B, T, HC), dtype=torch.bfloat16)
        k = torch.randn((B, T, HC), dtype=torch.bfloat16)
        v = torch.randn((B, T, HC), dtype=torch.bfloat16)
        kk = torch.randn((B, T, HC), dtype=torch.bfloat16)
        a = torch.randn((B, T, HC), dtype=torch.bfloat16)
        xx = torch.randn((B, T, HC), dtype=torch.bfloat16)

        wkv_state = torch.randn((B, NH, HS, HS), dtype=torch.float)

        xx_pt, wkv_state_pt = rwkv7_attn_pytorch.rwkv7_attn_pytorch(r, w, k, v, kk, a, B, T, NH, HS, xx, wkv_state)
        xx_ort, wkv_state_ort = rwkv7_attn_onnx.rwkv7_attn_onnx(r, w, k, v, kk, a, B, T, NH, HS, xx, wkv_state)

        self.assertTrue(torch.allclose(xx_pt, xx_ort, equal_nan=True))
        self.assertTrue(torch.allclose(wkv_state_pt, wkv_state_ort, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
