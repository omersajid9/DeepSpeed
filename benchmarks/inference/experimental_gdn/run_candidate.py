# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Opt-in process-local experiment. Never edits an engine checkout."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from experiment import ROOT, SOURCE_ROOT, build


def install(variant):
    import deepspeed.module_inject.segment_ki as seg
    from deepspeed.ops.op_builder import FusedGLUBuilder
    op = build()
    original_op = FusedGLUBuilder().load(verbose=False)
    counts = dict(norm=0, gdn=0)
    if variant in ("norm", "both", "triton_both", "triton_match"):
        old_norm = original_op.fused_add_norm

        def norm(h, r, w, eps, zero_centered):
            supported = (h.dtype == torch.bfloat16 and h.shape[-1] in (2560, 4096)
                         and all(x.is_contiguous() for x in (h, r, w))
                         and len({x.untyped_storage().data_ptr()
                                  for x in (h, r, w)}) == 3)
            if not supported or torch.is_grad_enabled():
                return old_norm(h, r, w, eps, zero_centered)
            out = torch.empty_like(h)
            op.add_norm(h, r, w, out, eps, zero_centered, 256)
            counts["norm"] += 1
            return out

        original_op.fused_add_norm = norm

    if variant in ("gdn_cuda4", "both", "triton_both", "triton_gdn", "triton_match", "triton_gdn_match"):

        def packed_step(parent, mixed, a, b, z, state, key_heads):
            if not counts["gdn"]:
                counts["gdn_state_dtype"] = str(state.dtype)
                counts["gdn_state_shape"] = list(state.shape)
                counts["gdn_state_stride"] = list(state.stride())
            counts["gdn"] += 1
            cached = getattr(parent, "_lab_gate_constants", None)
            if cached is None:
                cached = parent._lab_gate_constants = (parent.A_log.detach().float().contiguous(),
                                                       parent.dt_bias.detach().float().contiguous())
            batch = mixed.shape[0]
            out = torch.empty(batch, 1, state.shape[1], 128, device=mixed.device, dtype=mixed.dtype)
            if variant in ("triton_both", "triton_gdn", "triton_match", "triton_gdn_match"):
                from packed_triton import launch
                tile = 64 if state.dtype == torch.bfloat16 else 32
                launch(mixed[:, 0],
                       a[:, 0],
                       b[:, 0],
                       *cached,
                       state,
                       state,
                       out,
                       key_heads,
                       tile,
                       4,
                       match_fla=variant in ("triton_match", "triton_gdn_match"))
            else:
                op.packed_gdn(mixed[:, 0], a[:, 0], b[:, 0], *cached, state, state, out, key_heads, 4)
            core = out.reshape(-1, 128)
            z = z.reshape(-1, 128)
            if core.is_contiguous() and z.is_contiguous():
                normed = torch.empty_like(core)
                eps = getattr(parent.norm, "variance_epsilon", None) or parent.norm.eps
                original_op.gated_norm(core, z, parent.norm.weight, normed, eps)
            else:
                normed = parent.norm(core, z)
            return parent.out_proj(normed.reshape(batch, 1, -1))

        source = inspect.getsource(seg._fused_gdn_forward)
        marker = "    mixed_qkv = mixed_qkv.transpose(1, 2)\n"
        assert source.count(marker) == 1
        insertion = """
    # Experimental decode-only path; prefill/training/TP keep the original path.
    if (use_precomputed_states and seq_len == 1 and tp_size == 1
            and not torch.is_grad_enabled() and mixed_qkv.dtype is torch.bfloat16
            and self.head_k_dim == self.head_v_dim == 128
            and not cache_params.layers[self.layer_idx].record_past):
        lab_state = cache_params.layers[self.layer_idx].recurrent_states[0]
        if lab_state.is_contiguous() and lab_state.dtype in (torch.float32, torch.bfloat16):
            return _lab_packed_step(self, mixed_qkv, a, b, z, lab_state, num_k_heads)
"""
        source = source.replace(marker, marker + insertion)
        preview = ROOT / "forward_preview.py"
        preview.write_text("# SPDX-License-Identifier: Apache-2.0\n# DeepSpeed Team\n"
                           "# Experimental process-local forward, derived from segment_ki.py.\n" + source)
        namespace = dict(seg._fused_gdn_forward.__globals__, _lab_packed_step=packed_step)
        exec(compile(source, str(preview), "exec"), namespace)
        seg._fused_gdn_forward = namespace["_fused_gdn_forward"]
    return counts


def main():
    variant = os.environ["KERNEL_LAB_VARIANT"]
    assert variant in ("baseline", "norm", "gdn_cuda4", "both", "triton_both", "triton_gdn", "triton_match",
                       "triton_gdn_match")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    # The unmodified diagnostic generator imports its neighboring artifacts.py.
    sys.path.insert(0, str(Path(script).parent))
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {} if variant == "baseline" else install(variant)
    completed = False
    try:
        runpy.run_path(script, run_name="__main__")
        if variant in ("gdn_cuda4", "both", "triton_both", "triton_gdn", "triton_match", "triton_gdn_match"):
            assert counts["gdn"] > 0, "GDN candidate never dispatched; run is not a GDN validation"
        if variant in ("norm", "both", "triton_both", "triton_match"):
            assert counts["norm"] > 0, "Norm candidate never dispatched"
        completed = True
    finally:
        output.with_suffix(".candidate.json").write_text(
            json.dumps(dict(variant=variant,
                            completed=completed,
                            dispatch_calls=counts,
                            source_sha256=hashlib.sha256((SOURCE_ROOT / "kernels.cu").read_bytes()).hexdigest(),
                            triton_source_sha256=hashlib.sha256((SOURCE_ROOT /
                                                                 "packed_triton.py").read_bytes()).hexdigest(),
                            gpu=os.environ.get("CUDA_VISIBLE_DEVICES")),
                       indent=2) + "\n")


if __name__ == "__main__":
    main()
