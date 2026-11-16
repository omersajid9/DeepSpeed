# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Native [K,V] state alternative to the CUDA prototype, with the same gates."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def packed(qkv, a, b, a_log, bias, state, next_state, out, H: tl.constexpr, HV: tl.constexpr, QS: tl.constexpr,
           AS: tl.constexpr, BS: tl.constexpr, BV: tl.constexpr, MATCH_FLA_NORM: tl.constexpr,
           PRECOMPUTED_GATES: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HV, bh % HV
    qhead = head // (HV // H)
    ki = tl.arange(0, 128)
    vi = tile * BV + tl.arange(0, BV)
    q = tl.load(qkv + batch * QS + qhead * 128 + ki).to(tl.float32)
    k = tl.load(qkv + batch * QS + H * 128 + qhead * 128 + ki).to(tl.float32)
    if MATCH_FLA_NORM:
        # Same operation order as the installed FLA decode kernel.
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
        q = q * (128**-0.5)
    else:
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * (128**-0.5)
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    if PRECOMPUTED_GATES:
        beta = tl.load(a + batch * AS + head).to(tl.float32)
        g = tl.load(b + batch * BS + head).to(tl.float32)
    else:
        av = tl.load(a + batch * AS + head).to(tl.float32)
        bv = tl.load(b + batch * BS + head).to(tl.float32)
        x = av + tl.load(bias + head)
        # Preserve tiny decay gates rather than rounding log(1+tiny) to zero.
        softplus = tl.where(x > 20., x, tl.where(x < -10., libdevice.log1p(tl.exp(x)), tl.log(1. + tl.exp(x))))
        g = (-tl.exp(tl.load(a_log + head)) * softplus).to(tl.bfloat16).to(tl.float32)
        beta = tl.sigmoid(bv).to(tl.bfloat16).to(tl.float32)
    offset = bh * 128 * 128 + ki[:, None] * 128 + vi[None, :]
    s = tl.load(state + offset) * tl.exp(g)
    v = tl.load(qkv + batch * QS + 2 * H * 128 + head * 128 + vi).to(tl.float32)
    delta = (v - tl.sum(s * k[:, None], 0)) * beta
    s = s + k[:, None] * delta[None, :]
    y = tl.sum(s * q[:, None], 0)
    tl.store(next_state + offset, s)
    tl.store(out + bh * 128 + vi, y)


def launch(qkv, a, b, a_log, bias, state, next_state, out, H=16, tile=32, warps=4, match_fla=False, from_gates=False):
    batch, hv = state.shape[:2]
    packed[(128 // tile, batch * hv)](qkv,
                                      a,
                                      b,
                                      a_log,
                                      bias,
                                      state,
                                      next_state,
                                      out,
                                      H,
                                      hv,
                                      qkv.stride(0),
                                      a.stride(0),
                                      b.stride(0),
                                      tile,
                                      match_fla,
                                      from_gates,
                                      num_warps=warps,
                                      enable_fp_fusion=True)
