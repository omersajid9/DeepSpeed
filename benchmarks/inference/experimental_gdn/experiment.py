# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Independent-oracle checks and graph microbenchmarks; no engine changes."""
import csv
import json
import os
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from deepspeed.ops.op_builder import FusedGLUBuilder
from deepspeed.module_inject.segment_ki import _fla_recurrent_inplace
from packed_triton import launch as triton_gdn

SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KERNEL_LAB_OUTPUT", str(SOURCE_ROOT / "results"))).resolve()


def output_dir():
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def build():
    build_dir = output_dir() / "build"
    build_dir.mkdir(exist_ok=True)
    return load("b300_kernel_lab", [str(SOURCE_ROOT / "kernels.cu")],
                build_directory=str(build_dir),
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "--ptxas-options=-v"],
                verbose=True)


def error(actual, ref):
    diff = actual.double() - ref.double()
    return dict(relative_l2=(diff.norm() / ref.double().norm().clamp_min(1e-30)).item(),
                max_abs=diff.abs().max().item(),
                finite=bool(actual.isfinite().all()))


def timing(fn, count=64, repeats=7):
    # Captured launches remove Python dispatch gaps. These repeated single-layer
    # tests are cache-hot microbenchmarks, not model wall-time predictions.
    for _ in range(5):
        fn()
    torch.cuda.synchronize()  #ignore-cuda
    graph = torch.cuda.CUDAGraph()  #ignore-cuda
    with torch.cuda.graph(graph):  #ignore-cuda
        for _ in range(count):
            fn()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  #ignore-cuda
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / count)
    return dict(median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples))


def main():
    torch.manual_seed(271828)
    op, baseline = build(), FusedGLUBuilder().load(verbose=False)
    match_fla = os.environ.get("MATCH_FLA", "1") == "1"
    rows, checks = [], []
    state_dtype = torch.bfloat16 if os.environ.get("STATE_DTYPE") == "bf16" else torch.float32
    for dim in (() if state_dtype == torch.bfloat16 else (2560, 4096)):
        for batch in (2, 4, 8, 16, 32):
            h = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
            r, w = torch.randn_like(h), torch.randn(dim, device="cuda", dtype=torch.bfloat16) * .1
            original, out = h.clone(), torch.empty_like(h)
            for centered in (False, True):
                total = original.double() + r.double()
                ref = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)
                ref *= w.double() + int(centered)
                h.copy_(original)
                old = baseline.fused_add_norm(h, r, w, 1e-6, centered)
                for threads in (128, 256):
                    h.copy_(original)
                    op.add_norm(h, r, w, out, 1e-6, centered, threads)
                    metrics = error(out, ref)
                    assert metrics["finite"] and metrics["relative_l2"] < .005, metrics
                    assert torch.equal(h, (original + r)), "residual BF16 rounding changed"
                    checks.append(
                        dict(kind="norm",
                             dim=dim,
                             batch=batch,
                             centered=centered,
                             threads=threads,
                             oracle=metrics,
                             baseline_agreement=(old == out).float().mean().item()))
            # r=0 stabilizes the in-place stream across replay; nonzero residuals
            # were checked above. Every launch still reads and adds both operands.
            r.zero_()
            for name, fn in [
                ("existing", lambda: baseline.fused_add_norm(h, r, w, 1e-6, True)),
                ("cuda128", lambda: op.add_norm(h, r, w, out, 1e-6, True, 128)),
                ("cuda256", lambda: op.add_norm(h, r, w, out, 1e-6, True, 256)),
            ]:
                row = dict(kind="norm", dim=dim, batch=batch, variant=name, **timing(fn))
                rows.append(row)
                print(row, flush=True)

    H, HV, K, V = 16, 32, 128, 128
    for batch in (2, 4, 8, 16, 32):
        # Deliberately wider row strides catch incorrect assumptions about the
        # concatenated projection's qkv/a/b views.
        storage = torch.randn(batch, 2 * H * K + HV * V + HV * 2 + 64, device="cuda", dtype=torch.bfloat16)
        qkv = storage[:, :2 * H * K + HV * V]
        a = storage[:, -HV * 2:-HV]
        b = storage[:, -HV:]
        a_log = torch.randn(HV, device="cuda") * .2
        bias = torch.randn(HV, device="cuda")
        state = torch.randn(batch, HV, K, V, device="cuda", dtype=state_dtype) * .1
        out, next_state = torch.empty(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16), torch.empty_like(state)
        q, k, v = qkv.split([H * K, H * K, HV * V], -1)
        q, k, v = q.reshape(batch, 1, H, K), k.reshape(batch, 1, H, K), v.reshape(batch, 1, HV, V)
        beta, g = baseline.gdn_gates(a, b, a_log, bias)
        # FP64 recurrence, independent of the CUDA/Triton implementation.
        qref = q.squeeze(1).double().repeat_interleave(HV // H, 1)
        kref = k.squeeze(1).double().repeat_interleave(HV // H, 1)
        qref = qref / torch.sqrt(qref.square().sum(-1, keepdim=True) + 1e-6) / K**.5
        kref = kref / torch.sqrt(kref.square().sum(-1, keepdim=True) + 1e-6)
        # Match the public gate contract's BF16 intermediate storage.
        decayed = state.double() * g.double().exp()[..., None, None]
        delta = (v.squeeze(1).double() - torch.einsum("bhk,bhkv->bhv", kref, decayed)) * beta.double()[..., None]
        expected_state = decayed + kref[..., None] * delta[..., None, :]
        expected_out = torch.einsum("bhk,bhkv->bhv", qref, expected_state).unsqueeze(1)
        expected_state = expected_state.to(state_dtype).double()
        state_tolerance = .005 if state_dtype == torch.bfloat16 else 1e-5
        for warps in (4, 8):
            op.packed_gdn(qkv, a, b, a_log, bias, state, next_state, out, H, warps)
            om, sm = error(out, expected_out), error(next_state, expected_state)
            assert om["finite"] and sm["finite"] and om["relative_l2"] < .005 and sm[
                "relative_l2"] < state_tolerance, (om, sm)
            slot = state.clone()
            op.packed_gdn(qkv, a, b, a_log, bias, slot, slot, out, H, warps)
            assert torch.equal(slot, next_state), "in-place state update differs"
            checks.append(dict(kind="gdn", batch=batch, warps=warps, output=om, state=sm))
        for tile, warps in ((8, 1), (16, 2), (32, 4), (64, 4)):
            triton_gdn(qkv, a, b, a_log, bias, state, next_state, out, H, tile, warps, match_fla=match_fla)
            om, sm = error(out, expected_out), error(next_state, expected_state)
            assert om["finite"] and sm["finite"] and om["relative_l2"] < .005 and sm[
                "relative_l2"] < state_tolerance, (om, sm)
            checks.append(dict(kind="gdn_triton", batch=batch, tile=tile, warps=warps, output=om, state=sm))
        # Existing full chain includes gates, grouped Q/K materialization, and
        # the same FLA scan observed in the real model trace.
        slot = state.clone()

        def existing():
            be, ga = baseline.gdn_gates(a, b, a_log, bias)
            qr, kr, vr = baseline.repeat_qk(q, k, v, HV // H)
            return _fla_recurrent_inplace(qr, kr, vr, g=ga, beta=be, initial_state=slot, use_qk_l2norm_in_kernel=True)

        old, _ = existing()
        checks.append(
            dict(kind="gdn_existing_oracle",
                 batch=batch,
                 output=error(old, expected_out),
                 state=error(slot, expected_state)))
        for name, fn in [
            ("existing_chain", existing),
            ("cuda4", lambda: op.packed_gdn(qkv, a, b, a_log, bias, slot, slot, out, H, 4)),
            ("cuda8", lambda: op.packed_gdn(qkv, a, b, a_log, bias, slot, slot, out, H, 8)),
                *[(f"triton{tile}w{warps}", lambda tile=tile, warps=warps: triton_gdn(
                    qkv, a, b, a_log, bias, slot, slot, out, H, tile, warps, match_fla=match_fla))
                  for tile, warps in ((8, 1), (16, 2), (32, 4), (64, 4))],
        ]:
            slot.copy_(state)
            row = dict(kind="gdn", dim=K, batch=batch, variant=name, **timing(fn))
            rows.append(row)
            print(row, flush=True)
    suffix = "_bf16" if state_dtype == torch.bfloat16 else ""
    (ROOT / f"correctness{suffix}.json").write_text(json.dumps(checks, indent=2) + "\n")
    with (ROOT / f"microbench{suffix}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / f"environment{suffix}.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),  #ignore-cuda
                capability=torch.cuda.get_device_capability(),  #ignore-cuda
                seed=271828,
                method="CUDA graph 64 calls; 7 repeats; same GPU sequentially"),
            indent=2) + "\n")


if __name__ == "__main__":
    with torch.no_grad():
        main()
