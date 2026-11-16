# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Long recurrence, gate extremes, and state working sets larger than L2."""
import json
import os

import torch
from deepspeed.ops.op_builder import FusedGLUBuilder
from deepspeed.module_inject.segment_ki import _fla_recurrent_inplace
from experiment import ROOT, build, error, timing
from packed_triton import launch


@torch.no_grad()
def main():
    torch.manual_seed(1234567)
    op, base = build(), FusedGLUBuilder().load(verbose=False)
    checks, times = [], []
    dtype = torch.bfloat16 if os.environ.get("STATE_DTYPE", "bf16") == "bf16" else torch.float32
    suffix = "bf16" if dtype == torch.bfloat16 else "fp32"
    match_fla = os.environ.get("MATCH_FLA", "1") == "1"
    if match_fla:
        suffix += "_match"
    # Vary a/b across sigmoid saturation, the softplus branches, and g≈0.
    B, H, HV = 2, 16, 32
    qkv = torch.randn(B, 8192, device="cuda", dtype=torch.bfloat16)
    a = torch.linspace(-80, 80, B * HV, device="cuda").reshape(B, HV).bfloat16()
    b = a.flip(-1).contiguous()
    a_log, bias = torch.zeros(HV, device="cuda"), torch.zeros(HV, device="cuda")
    state = torch.randn(B, HV, 128, 128, device="cuda", dtype=dtype) * .1
    slots = {name: state.clone() for name in ("cuda4", "triton32", "triton64")}
    outs = {name: torch.empty(B, 1, HV, 128, device="cuda", dtype=torch.bfloat16) for name in slots}
    ref = state.double()
    for step in range(1024):
        qkv.normal_()
        beta, g = base.gdn_gates(a, b, a_log, bias)
        q, k, v = qkv.double().split((2048, 2048, 4096), -1)
        q = q.reshape(B, H, 128).repeat_interleave(2, 1)
        k = k.reshape(B, H, 128).repeat_interleave(2, 1)
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**.5
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ref *= g.double().exp()[..., None, None]
        delta = (v.reshape(B, HV, 128) - torch.einsum("bhk,bhkv->bhv", k, ref)) * beta.double()[..., None]
        ref += k[..., None] * delta[..., None, :]
        expected = torch.einsum("bhk,bhkv->bhv", q, ref).unsqueeze(1)
        ref = ref.to(dtype).double()
        op.packed_gdn(qkv, a, b, a_log, bias, slots["cuda4"], slots["cuda4"], outs["cuda4"], H, 4)
        launch(qkv,
               a,
               b,
               a_log,
               bias,
               slots["triton32"],
               slots["triton32"],
               outs["triton32"],
               H,
               32,
               4,
               match_fla=match_fla)
        launch(qkv,
               a,
               b,
               a_log,
               bias,
               slots["triton64"],
               slots["triton64"],
               outs["triton64"],
               H,
               64,
               4,
               match_fla=match_fla)
        if step in (0, 31, 127, 511, 1023):
            for name in slots:
                om, sm = error(outs[name], expected), error(slots[name], ref)
                state_limit = .01 if dtype == torch.bfloat16 else 1e-5
                output_limit = .01 if dtype == torch.bfloat16 else .005
                assert om["finite"] and sm["finite"] and om["relative_l2"] < output_limit and sm[
                    "relative_l2"] < state_limit, (name, step, om, sm)
                checks.append(dict(step=step + 1, variant=name, output=om, state=sm))
    (ROOT / f"long_recurrence_{suffix}.json").write_text(json.dumps(checks, indent=2) + "\n")
    for B in (2, 32):
        qkv = torch.randn(B, 8192, device="cuda", dtype=torch.bfloat16)
        a, b = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16), torch.randn(B,
                                                                                    HV,
                                                                                    device="cuda",
                                                                                    dtype=torch.bfloat16)
        # 24 distinct layers: 96 MiB at b2, 1.5 GiB at b32. Report the size
        # explicitly; do not equate this synthetic working set to model timing.
        pool = [torch.randn(B, HV, 128, 128, device="cuda", dtype=dtype) * .1 for _ in range(24)]
        out = torch.empty(B, 1, HV, 128, device="cuda", dtype=torch.bfloat16)
        q, k, v = qkv.split((2048, 2048, 4096), -1)
        q, k, v = q.reshape(B, 1, H, 128), k.reshape(B, 1, H, 128), v.reshape(B, 1, HV, 128)
        for variant in ("existing_chain", "cuda4", "triton32", "triton64"):
            index = 0

            def call():
                nonlocal index
                slot = pool[index % len(pool)]
                index += 1
                if variant == "existing_chain":
                    beta, g = base.gdn_gates(a, b, a_log, bias)
                    qr, kr, vr = base.repeat_qk(q, k, v, 2)
                    _fla_recurrent_inplace(qr,
                                           kr,
                                           vr,
                                           g=g,
                                           beta=beta,
                                           initial_state=slot,
                                           use_qk_l2norm_in_kernel=True)
                elif variant == "cuda4":
                    op.packed_gdn(qkv, a, b, a_log, bias, slot, slot, out, H, 4)
                else:
                    launch(qkv,
                           a,
                           b,
                           a_log,
                           bias,
                           slot,
                           slot,
                           out,
                           H,
                           64 if variant == "triton64" else 32,
                           4,
                           match_fla=match_fla)

            row = dict(batch=B,
                       variant=variant,
                       state_dtype=str(dtype),
                       state_bytes=sum(s.numel() * s.element_size() for s in pool),
                       **timing(call, count=48))
            times.append(row)
            print(row, flush=True)
    (ROOT / f"streaming_bench_{suffix}.json").write_text(json.dumps(times, indent=2) + "\n")


if __name__ == "__main__":
    main()
