# OPSD continuous-batch + graph rollout baseline (1 GPU, Qwen3-4B/8B, Slurm/Delta)

Discovery notes only. No GPU jobs were run while preparing this document.

The baseline in this runbook is **continuous batching + graph capture**, not the static eager path. It is selected by:

- `SamplingConfig.continuous_batch_size` (continuous batching)
- `HybridEngineRolloutConfig.use_graph_capture` (graph capture)

Main branch already contains the implementation and tests. What it does not provide is a default trainer/benchmark entry point that selects them, so the OPSD trainer needs a small patch and the rollout-only baseline uses a standalone script.

## Where to run what

- **Login node, no GPU allocation:** sections 2–5 (environment install, data download/conversion, config files, OPSD patch) are CPU/IO-only. On a shared Lustre filesystem the compute node sees the same files.
- **GPU node, `srun` shell:** section 6 (one training step) and section 7 (standalone rollout baseline) are the only parts that need the GPU allocation.
- Keep `HF_HOME` and the conda env on `/work/nvme/bcjw/$USER` (or `/work/hdd/bcjw/$USER`). The `DeepSpeedExamples` checkout can stay in home; it is small. The one-step `runs/` output is tiny and can stay next to the repo; for longer runs, point `training.save_dir` at `/work/...`. Do not use `/work/$USER` (it does not exist) or `/projects/bcjw` (full). Avoid node-local `/tmp`.
- The step-2 build only needs `DS_BUILD_CPU_ADAM=1`, which is a CPU extension and builds without a GPU. If login-node policy forbids compilation, run the same `pip install` in a CPU-only Slurm job or on the GPU node before the timed run.
- On a GPU-less login node, `torch.cuda.is_available()` is expected to be `False`; that is not a broken install. Verify `import deepspeed, transformers` instead. If you do build CUDA extensions such as FusedAdam on a GPU-less node, set `TORCH_CUDA_ARCH_LIST="8.0;8.6"` (A100/A40) first.

## TL;DR

1. On the login node, create the conda env and install DeepSpeed + `training/opsd` requirements (section 2).
2. Download/convert ReTool-SFT (decode-heavy) or LongBench (prefill-heavy) into `data/*.jsonl` (section 3).
3. Create the configs and apply the `training/opsd` patch (sections 4–5).
4. Start the interactive 1-GPU `srun` shell (section 1) only now.
5. Run one training step with `use_graph_capture: true` and `continuous_batch_size: 2` (section 6).
6. For a clean steady-state rollout baseline (graph captured once, then replayed), run the standalone script in section 7.

---

## What exists in the repo

- Core `DeepSpeed` contains the rollout engine: `deepspeed/runtime/rollout/` (`HybridEngineRollout`, `RolloutConfig`, `SamplingConfig`, OPSD loss primitives). There is **no trainer, dataset, or task registry** in the core repo.
- The runnable 1-step trainer and rollout benchmark live in `deepspeedai/DeepSpeedExamples`:
  - `training/opsd/` — on-policy distillation trainer (rollout -> teacher forward -> student forward/backward).
  - `benchmarks/opsd/benchmark_hybrid_engine_rollout.py` — pure rollout benchmark; it is **static-only** and does not pass `continuous_batch_size`.
- Main branch has continuous batching + graph:
  - `SamplingConfig.continuous_batch_size` selects continuous batching.
  - `HybridEngineRolloutConfig.use_graph_capture` selects graph capture.
  - Commits on `master`: `e5887a35e` (continuous prototype), `8950dc332` (continuous profiling), `a77aeb676` (align/trim), `0eb3a8bcc` (CUDA graph for continuous rollouts).
  - Tests: `tests/unit/runtime/rollout/test_hybrid_engine_rollout.py` and `tests/unit/v1/test_hybrid_engine_rollout_v1.py::test_continuous_graph_generation_refills_a_fixed_slot_on_cuda`.
- Qwen3 is **not** in DeepSpeed's HybridEngine v1 inference policy list (`deepspeed/module_inject/replace_policy.py`). The static path falls back to HF `generate()`. The continuous path uses `HybridEngineRollout`'s own scheduler + `DeepSpeedStaticCache` + graph capture, so it does not depend on the DS inference-kernel policies.
- OPSD is single-turn prompt -> generate. It has no tool sandbox, so ReTool-SFT can supply agentic-style prompts/long responses, but it will not execute code or feed `<interpreter>` output back.

---

## 1. Slurm shell (1 GPU)

**Do this last.** Sections 2–5 run on the login node. Start this shell only for sections 6–7.

Reference

```
srun -A bcjw-delta-gpu --time=00:30:00 --nodes=1 --ntasks-per-node=16 --partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=32g --pty /bin/bash
```

Use your format; bump time/memory:

```bash
srun -A bcjw-delta-gpu --time=04:00:00 --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=128g --pty /bin/bash
```

For Qwen3-8B student + teacher, use `--mem=256g --time=06:00:00`.

Use `--ntasks=1 --cpus-per-task=16`. Do **not** use `--ntasks-per-node=16` for a command that itself launches DeepSpeed: Slurm runs 16 copies of the command, each `deepspeed --num_gpus 1` tries to bind port 29500, and 15 die with `EADDRINUSE`. `--ntasks-per-node=16` is only appropriate when Slurm itself is launching 16 ranks.

If your site rejects `--gpus=1`, use `--gpus-per-node=1`.

### `EADDRINUSE` on port 29500

If you see:

```text
torch.distributed.DistNetworkError: ... port: 29500 ... EADDRINUSE ...
```

the launcher ran multiple copies of DeepSpeed instead of one. Typical cause: a batch script with `srun --ntasks-per-node=16 ... bash -lc 'deepspeed --num_gpus 1 ...'`. Each task starts its own DeepSpeed launcher and they race for the rendezvous port.

Fix the topology:

```bash
# one DeepSpeed launcher, one GPU, 16 CPU cores
srun ... --ntasks=1 --cpus-per-task=16 --gpus=1 ... \
  bash -lc 'deepspeed --num_gpus 1 --master_port 29500 main.py ...'
```

Do not add `--master_port=$((29500 + SLURM_PROCID))` unless you actually intend independent single-GPU jobs. Changing the port hides the topology bug and produces N independent jobs instead of one.

Do not start this shell until you finish sections 2–5 on the login node. Model/dataset downloads happen in section 2, after the environment exists; compute nodes often have no outbound internet.

---

## 2. Environment setup (login node, no GPU)

Run sections 2–5 on the login node. They need CPU/disk, not a GPU.

```bash
# /u is NFS with a 100 GB quota. Keep the env and caches on Lustre.
cat > ~/opsd_env.sh <<'SH'
export WORK=/work/nvme/bcjw/$USER
export HF_HOME="$WORK/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# CPUAdam is built with GCC 14; the system libstdc++ is too old.
export LD_LIBRARY_PATH="$WORK/conda/opsd/lib:${LD_LIBRARY_PATH:-}"
SH
source ~/opsd_env.sh
mkdir -p "$HF_HOME" "$WORK/conda"

module load anaconda3_gpu 2>/dev/null || true
# If conda activate complains that the shell is not initialized:
# eval "$(conda shell.bash hook)"

# -p keeps the conda env off the NFS home quota.
conda create -p "$WORK/conda/opsd" python=3.12 -y
conda activate "$WORK/conda/opsd"

# CUDA-matched torch; adjust cu126 if your Delta image differs
pip install torch --index-url https://download.pytorch.org/whl/cu126

if [ ! -d ~/DeepSpeedExamples ]; then
  git clone https://github.com/deepspeedai/DeepSpeedExamples.git ~/DeepSpeedExamples
fi
cd ~/DeepSpeedExamples/training/opsd

# CPUAdam is required for the ZeRO-1 optimizer offload used by this baseline.
# CPUAdam is a CPU extension, so it builds on a GPU-less login node.
DS_BUILD_OPS=0 DS_BUILD_CPU_ADAM=1 \
  pip install -e /u/msajid/projects/DeepSpeed --no-build-isolation

pip install -r requirements.txt

# transformers/datasets/huggingface_hub are installed now, so these work.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B")
snapshot_download("JoeYing/ReTool-SFT", repo_type="dataset")
PY

python -c "import deepspeed, transformers; print(deepspeed.__version__, transformers.__version__)"
# torch.cuda.is_available() is False on a GPU-less login node; that is expected.
```

`offload_optimizer` uses `DeepSpeedCPUAdam`, which needs `DS_BUILD_CPU_ADAM=1` (or JIT at first use). CPUAdam needs a C++ compiler (load your site's GCC module if needed), not a GPU or `nvcc`.

The `LD_LIBRARY_PATH` line in `~/opsd_env.sh` is required on this login node: CPUAdam is compiled with GCC 14 and needs `GLIBCXX_3.4.30`, while `/lib64/libstdc++.so.6` is older. The conda env ships `libstdc++.so.6.0.34`; pointing `LD_LIBRARY_PATH` at `$WORK/conda/opsd/lib` fixes the import. Without it you get `version 'GLIBCXX_3.4.30' not found` when constructing `DeepSpeedCPUAdam`.

Before section 6, in the `srun` shell:

```bash
source ~/opsd_env.sh
module load anaconda3_gpu 2>/dev/null || true
conda activate "$WORK/conda/opsd"
```

---

## 3. Download/convert data

Run these in the same login-node shell (or re-source the env):

```bash
source ~/opsd_env.sh
conda activate "$WORK/conda/opsd"
```

### Option A — ReTool-SFT (agentic-style, decode-heavy)

```bash
cd ~/DeepSpeedExamples/training/opsd
mkdir -p data

python - <<'PY'
import json
from datasets import load_dataset

ds = load_dataset("JoeYing/ReTool-SFT", split="train")
with open("data/retool_prompts.jsonl", "w") as f:
    for row in ds:
        prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
        f.write(json.dumps({"prompt": prompt}) + "\n")
print(f"wrote {len(ds)} prompts")
PY

wc -l data/retool_prompts.jsonl
```

This is a tool-use-style SFT trace; the model will generate `<code>` / `<interpreter>` blocks but the trainer will not execute them.

### Option B — LongBench (long-context, prefill-heavy)

`datasets>=5.0` no longer runs dataset scripts, so
`load_dataset("zai-org/LongBench", ...)` fails with
`Dataset scripts are no longer supported`. Use the official data zip:

```bash
mkdir -p /tmp/longbench && cd /tmp/longbench
curl -L --fail -o longbench.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip -q -o longbench.zip
ls data/multifieldqa_en.jsonl

# Convert the zip's JSONL to the OPSD prompt format.
cd ~/DeepSpeedExamples/training/opsd
mkdir -p data
python - <<'PY'
import json

TEMPLATE = (
    "Read the following text and answer the question. "
    "Answer concisely.\n\n{context}\n\nQuestion: {input}\nAnswer:"
)
src = "/tmp/longbench/data/multifieldqa_en.jsonl"
dst = "data/longbench_mfqa.jsonl"
with open(src) as f, open(dst, "w") as out:
    for line in f:
        r = json.loads(line)
        out.write(json.dumps({
            "prompt": TEMPLATE.format(context=r["context"][-24000:], input=r["input"])
        }) + "\n")
print(f"wrote {dst}")
PY
```

### Task menu

| Task | Source | Prompt length | Decode length | Use |
|---|---|---|---|---|
| Bundled math | `training/opsd/data/prompts.jsonl` | ~20 tok | ~50 tok | Smoke only |
| ReTool-SFT | `JoeYing/ReTool-SFT` | ~0.3–1k tok | ~1–8k tok | Agentic-style, decode-heavy; **primary baseline** |
| LongBench `multifieldqa_en` | `zai-org/LongBench` | ~4–6k tok | ~50–200 tok | Long-context, prefill-heavy |
| LongBench `hotpotqa` | same | ~9–12k tok | short | Prefill-heavy; truncate to 8192 |
| True ReTool closed loop | verl `recipe/retool` | multi-turn | long | Needs tool sandbox; **not** this trainer |

---

## 4. Configs

### `configs/baseline_zero1_offload.json`

Student uses ZeRO-1 + CPU optimizer offload, so parameters stay on GPU (no per-forward gather), which is what makes graph capture viable. The teacher is still offloaded separately by `TeacherWrapper`.

From `~/DeepSpeedExamples/training/opsd`:

```bash
source ~/opsd_env.sh
conda activate "$WORK/conda/opsd"
cd ~/DeepSpeedExamples/training/opsd
mkdir -p configs

cat > configs/baseline_zero1_offload.json <<'JSON'
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": { "device": "cpu", "pin_memory": true }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-6,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.0
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": { "warmup_min_lr": 0, "warmup_max_lr": 1e-6, "warmup_num_steps": 1 }
  },
  "gradient_clipping": 1.0,
  "hybrid_engine": {
    "enabled": true,
    "max_out_tokens": 12288,
    "inference_tp_size": 1,
    "release_inference_cache": false,
    "pin_parameters": true,
    "tp_gather_partition_size": 8
  },
  "wall_clock_breakdown": true
}
JSON
```

### `configs/baseline_qwen3_4b_continuous_graph.json`

```bash
cat > configs/baseline_qwen3_4b_continuous_graph.json <<'JSON'
{
  "student": {
    "model_name_or_path": "Qwen/Qwen3-4B",
    "dtype": "bfloat16",
    "trust_remote_code": false
  },
  "teacher": {
    "model_name_or_path": "Qwen/Qwen3-4B",
    "dtype": "bfloat16",
    "trust_remote_code": false,
    "offload_to_cpu": true
  },
  "rollout": {
    "engine": "hybrid_engine",
    "use_graph_capture": true,
    "enable_profiling": true,
    "continuous_batch_size": 2,
    "max_prompt_length": 2048,
    "max_response_length": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "n_samples_per_prompt": 1
  },
  "distillation": {
    "loss_type": "reverse_kl",
    "temperature": 1.0,
    "chunk_size": 512
  },
  "training": {
    "micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "num_train_epochs": 1,
    "max_steps": 1,
    "warmup_steps": 0,
    "save_steps": 100000,
    "logging_steps": 1,
    "save_dir": "./runs/qwen3_4b_continuous_graph",
    "seed": 42
  },
  "data": {
    "path": "data/retool_prompts.jsonl",
    "prompt_field": "prompt",
    "shuffle": false
  },
  "deepspeed_config": "configs/baseline_zero1_offload.json"
}
JSON
```

`micro_batch_size_per_gpu` must be larger than `continuous_batch_size` so the scheduler has pending requests to refill. `temperature: 0` and `n_samples_per_prompt: 1` are required by the continuous path.

If the 4B student OOMs at `4 / 2`, drop to `micro_batch_size_per_gpu: 2` and `continuous_batch_size: 1`. The scheduler still refills the single slot after the first request retires, but there is no decode overlap.

### LongBench variant

Create it from the 4B config:

```bash
python - <<'PY'
import json

src = "configs/baseline_qwen3_4b_continuous_graph.json"
dst = "configs/baseline_qwen3_4b_longbench_continuous_graph.json"
c = json.load(open(src))
c["rollout"]["max_prompt_length"] = 8192
c["rollout"]["max_response_length"] = 256
c["rollout"]["continuous_batch_size"] = 1
c["training"]["micro_batch_size_per_gpu"] = 2
c["data"]["path"] = "data/longbench_mfqa.jsonl"
c["training"]["save_dir"] = "./runs/qwen3_4b_longbench_continuous_graph"
json.dump(c, open(dst, "w"), indent=2)
print(f"wrote {dst}")
PY
```

With `continuous_batch_size: 1` and two requests, the scheduler still refills the single slot after the first request finishes, but there is no decode overlap. Use `4 / 2` only if memory allows at 8k prompt length.

For Qwen3-8B, create the variant and raise Slurm memory to `256g`:

```bash
python - <<'PY'
import json

src = "configs/baseline_qwen3_4b_continuous_graph.json"
dst = "configs/baseline_qwen3_8b_continuous_graph.json"
c = json.load(open(src))
c["student"]["model_name_or_path"] = "Qwen/Qwen3-8B"
c["teacher"]["model_name_or_path"] = "Qwen/Qwen3-8B"
c["training"]["micro_batch_size_per_gpu"] = 2
c["rollout"]["continuous_batch_size"] = 1
c["training"]["save_dir"] = "./runs/qwen3_8b_continuous_graph"
json.dump(c, open(dst, "w"), indent=2)
print(f"wrote {dst}")
PY
```

Also download the 8B weights if you have not already:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

Qwen3-8B params + gradients are ~32 GB before activations, so a 40 GB A100 will likely OOM even with optimizer offload. Use an A40 48 GB and batch 2 / continuous 1, or prefer Qwen3-4B for the first graph-capture baseline. Do not add `offload_param` to the student and then expect graph capture to be the validated path.

---

## 5. One-time patch: enable continuous batching + profile output in OPSD

`training/opsd/config.py` does not accept `enable_profiling` or `continuous_batch_size`, and `trainer.py` never passes `continuous_batch_size` or reads `get_last_profile()`. Run this from `~/DeepSpeedExamples/training/opsd` in the activated env:

```bash
source ~/opsd_env.sh
conda activate "$WORK/conda/opsd"
cd ~/DeepSpeedExamples/training/opsd

python - <<'PY'
from pathlib import Path

cfg = Path("config.py")
s = cfg.read_text()
if "enable_profiling" not in s:
    s = s.replace(
        "    n_samples_per_prompt: int = 1\n",
        "    n_samples_per_prompt: int = 1\n    enable_profiling: bool = False\n",
    )
if "continuous_batch_size" not in s:
    s = s.replace(
        "    n_samples_per_prompt: int = 1\n",
        "    n_samples_per_prompt: int = 1\n    continuous_batch_size: Optional[int] = None\n",
    )
cfg.write_text(s)

tr = Path("trainer.py")
s = tr.read_text()
if "import json" not in s:
    s = s.replace("import os\nimport time\n", "import json\nimport os\nimport time\n")

if "continuous_batch_size=self.cfg.rollout.continuous_batch_size" not in s:
    s = s.replace(
        "            n_samples_per_prompt=self.cfg.rollout.n_samples_per_prompt,\n",
        "            n_samples_per_prompt=self.cfg.rollout.n_samples_per_prompt,\n"
        "            continuous_batch_size=self.cfg.rollout.continuous_batch_size,\n",
    )

if "rollout_profile =" not in s:
    s = s.replace(
        "        t_rollout = time.time() - t_start\n",
        "        rollout_profile = self.rollout.get_last_profile()\n"
        "        continuous_stats = self.rollout.get_last_continuous_stats()\n"
        "        t_rollout = time.time() - t_start\n",
    )

if '"rollout_profile": rollout_profile' not in s:
    s = s.replace(
        '            "response_tokens": int(mask_shifted.sum().item()),\n',
        '            "response_tokens": int(mask_shifted.sum().item()),\n'
        '            "rollout_profile": rollout_profile,\n'
        '            "continuous_stats": continuous_stats,\n'
        '            "peak_mem_mb": get_accelerator().max_memory_allocated() / (1024**2),\n',
    )

# Upgrade a file that already has the older profiling patch but not the
# continuous-batching fields.
if "continuous_stats = self.rollout.get_last_continuous_stats()" not in s:
    s = s.replace(
        "        rollout_profile = self.rollout.get_last_profile()\n",
        "        rollout_profile = self.rollout.get_last_profile()\n"
        "        continuous_stats = self.rollout.get_last_continuous_stats()\n",
    )

if '"continuous_stats": continuous_stats' not in s:
    s = s.replace(
        '            "rollout_profile": rollout_profile,\n',
        '            "rollout_profile": rollout_profile,\n'
        '            "continuous_stats": continuous_stats,\n',
    )

if "reset_peak_memory_stats" not in s:
    s = s.replace(
        "        t_start = time.time()\n",
        "        get_accelerator().reset_peak_memory_stats()\n        t_start = time.time()\n",
        1,
    )

if "step_metrics.jsonl" not in s:
    old = """    def _maybe_log(self, metrics: dict) -> None:
        if self.step % self.cfg.training.logging_steps != 0:
"""
    new = """    def _maybe_log(self, metrics: dict) -> None:
        os.makedirs(self.cfg.training.save_dir, exist_ok=True)
        metrics_path = os.path.join(self.cfg.training.save_dir, "step_metrics.jsonl")
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": self.step, **metrics}) + "\\n")

        if self.step % self.cfg.training.logging_steps != 0:
"""
    s = s.replace(old, new)

tr.write_text(s)
print("patched config.py and trainer.py")
PY
```

Check the patched files:

```bash
python -m py_compile config.py trainer.py
```

This writes `runs/<run>/step_metrics.jsonl` with one JSON object per step, including `rollout_profile` and `continuous_stats`.

---

## 6. Run one training step with continuous batching + graph

In the `srun` shell:

```bash
source ~/opsd_env.sh
module load anaconda3_gpu 2>/dev/null || true
conda activate "$WORK/conda/opsd"
cd ~/DeepSpeedExamples/training/opsd
mkdir -p runs

deepspeed --num_gpus 1 --master_port 29500 main.py \
  --config configs/baseline_qwen3_4b_continuous_graph.json \
  2>&1 | tee runs/qwen3_4b_continuous_graph.stdout.log
```

Expected stdout:

```text
[opsd][step 0] loss=... rollout=...s teacher=...s student=...s step=...s resp_tok=...
```

The JSONL contains:

```text
runs/qwen3_4b_continuous_graph/step_metrics.jsonl
```

Read it:

```bash
python - <<'PY'
import json

for line in open("runs/qwen3_4b_continuous_graph/step_metrics.jsonl"):
    r = json.loads(line)
    p = r["rollout_profile"] or {}
    s = r["continuous_stats"] or {}
    print(f"step={r['step']} loss={r['loss']:.4f} "
          f"rollout={r['rollout_s']:.2f}s teacher={r['teacher_s']:.2f}s "
          f"student={r['student_s']:.2f}s step={r['step_s']:.2f}s")
    print(f"  prefill={p.get('prefill_forward_ms')}ms "
          f"decode={p.get('decode_forward_ms')}ms "
          f"prefill_fwds={p.get('num_prefill_forwards')} "
          f"decode_fwds={p.get('num_decode_forwards')} "
          f"scheduler={p.get('scheduler_overhead_ms')}ms "
          f"cache={p.get('cache_management_overhead_ms')}ms "
          f"active_bs={p.get('active_batch_size')} "
          f"cont_bs={p.get('continuous_batch_size')} "
          f"tok/s={p.get('tokens_per_second')} "
          f"peak_mem={r.get('peak_mem_mb', 0):.0f}MB")
    print(f"  stats={s}")
PY
```

Notes:

- Graph capture happens on the first generation. With `max_steps=1`, the capture cost is included in the only measured step. The graph signature includes parameter `data_ptr()` and `_version`, so an optimizer update that changes either forces a recapture; expect a recapture after `step()`. This is expected; continuous+graph is a generation-path optimization, not a free per-step win.
- For steady-state graph-replay numbers, use the standalone script in section 7. It captures once during warmup and then measures replay iterations without weight updates.
- `enable_profiling=True` makes `rollout.generate()` synchronize at the boundaries, so `rollout_s` is real wall time.
- `rollout_s` isolates generation. `teacher_s` and `student_s` are separate; do not use `step_s` as the rollout baseline.

---

## 7. Clean rollout-only baseline: continuous batching + graph

This is the most reliable way to get steady-state continuous+graph numbers. It uses the raw model through `HybridEngineRollout`, not the OPSD trainer, so there is no optimizer/ZeRO interaction. It still runs on one GPU and uses real prompts.

Create `benchmarks/bench_continuous_graph.py` from `~/DeepSpeedExamples/training/opsd`:

```bash
source ~/opsd_env.sh
conda activate "$WORK/conda/opsd"
cd ~/DeepSpeedExamples/training/opsd

cat > benchmarks/bench_continuous_graph.py <<'PY'
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Continuous-batching + graph-capture rollout baseline for one GPU."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)


def _load_request(path, batch_size, tokenizer, max_prompt_length, device):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) == batch_size:
                break
    if len(prompts) != batch_size:
        raise ValueError(f"need {batch_size} prompts, found {len(prompts)}")

    encoded = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded.append(
            tokenizer(rendered, add_special_tokens=False, truncation=True,
                      max_length=max_prompt_length)["input_ids"])

    width = max(len(ids) for ids in encoded)
    pad = tokenizer.pad_token_id
    prompt_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    prompt_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    for row, ids in enumerate(encoded):
        n = len(ids)
        prompt_ids[row, width - n:] = torch.tensor(ids, dtype=torch.long)
        prompt_mask[row, width - n:] = 1

    return RolloutRequest(prompt_ids.to(device), prompt_mask.to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--continuous-batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", default="runs/continuous_graph_baseline.json")
    args = parser.parse_args()

    device = torch.device(get_accelerator().current_device_name())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    request = _load_request(args.data, args.batch_size, tokenizer,
                            args.max_prompt_length, device)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(use_graph_capture=True, enable_profiling=True),
    )
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        n_samples_per_prompt=1,
        continuous_batch_size=args.continuous_batch_size,
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            rollout.generate(request, sampling)
        get_accelerator().synchronize()
        get_accelerator().reset_peak_memory_stats()

        iterations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            rollout.generate(request, sampling)
            get_accelerator().synchronize()
            wall_ms = (time.perf_counter() - start) * 1000.0
            iterations.append({
                "wall_ms": wall_ms,
                "profile": dict(rollout.get_last_profile() or {}),
                "continuous_stats": dict(rollout.get_last_continuous_stats() or {}),
            })

    result = {
        "model": args.model,
        "data": args.data,
        "batch_size": args.batch_size,
        "continuous_batch_size": args.continuous_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "graph_capture": True,
        "peak_memory_mb": get_accelerator().max_memory_allocated() / (1024**2),
        "iterations": iterations,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
PY
```

Check it:

```bash
python -m py_compile benchmarks/bench_continuous_graph.py
```

Run the ReTool decode-heavy baseline:

```bash
cd ~/DeepSpeedExamples/training/opsd
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B \
  --data data/retool_prompts.jsonl \
  --batch-size 4 \
  --continuous-batch-size 2 \
  --max-prompt-length 2048 \
  --max-new-tokens 512 \
  --warmup 1 \
  --iterations 3 \
  --output runs/qwen3_4b_retool_continuous_graph.json
```

Run the LongBench prefill-heavy baseline:

```bash
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B \
  --data data/longbench_mfqa.jsonl \
  --batch-size 2 \
  --continuous-batch-size 1 \
  --max-prompt-length 8192 \
  --max-new-tokens 256 \
  --warmup 1 \
  --iterations 3 \
  --output runs/qwen3_4b_longbench_continuous_graph.json
```

Read the steady-state profile:

```bash
python - <<'PY'
import json

p = json.load(open("runs/qwen3_4b_retool_continuous_graph.json"))
print(f"model={p['model']} batch={p['batch_size']} cont={p['continuous_batch_size']} "
      f"peak={p['peak_memory_mb']:.0f}MB")
for i, it in enumerate(p["iterations"]):
    prof = it["profile"]
    stats = it["continuous_stats"]
    print(f"iter={i} wall={it['wall_ms']:.1f}ms "
          f"prefill={prof.get('prefill_forward_ms'):.1f}ms "
          f"decode={prof.get('decode_forward_ms'):.1f}ms "
          f"decode_fwds={prof.get('num_decode_forwards')} "
          f"scheduler={prof.get('scheduler_overhead_ms'):.1f}ms "
          f"cache={prof.get('cache_management_overhead_ms'):.1f}ms "
          f"tok/s={prof.get('tokens_per_second'):.1f}")
    print(f"  stats={stats}")
PY
```

### Optional static reference

`benchmarks/opsd/benchmark_hybrid_engine_rollout.py` remains useful as a static-eager reference, but it does **not** select continuous batching or graph capture. Keep it only for before/after comparison:

```bash
cd ~/DeepSpeedExamples
PYTHONPATH=/u/msajid/projects/DeepSpeed:$PWD \
torchrun --nproc_per_node=1 --master_port 29501 \
  benchmarks/opsd/benchmark_hybrid_engine_rollout.py \
  --model Qwen/Qwen3-4B --dtype bf16 \
  --batch-sizes 1 --samples-per-prompt 1 \
  --prompt-lengths 2048 8192 --response-lengths 256 1024 \
  --warmup 1 --iterations 3 \
  --output runs/qwen3_4b_static_reference.json
```

---

## 8. Knobs for the continuous + graph baseline

| Knob | Where | Value | Why |
|---|---|---|---|
| `rollout.continuous_batch_size` | OPSD config (after patch) | `2` | Selects continuous batching; scheduler slot count |
| `rollout.use_graph_capture` | OPSD config (base `RolloutConfig`) | `true` | Selects `_generate_continuous_graph` |
| `rollout.enable_profiling` | OPSD config (after patch) | `true` | Emits prefill/decode/scheduler/cache metrics |
| `rollout.temperature` | OPSD config | `0.0` | Continuous path is greedy-only |
| `rollout.n_samples_per_prompt` | OPSD config | `1` | Continuous path requires one sample per prompt |
| `training.micro_batch_size_per_gpu` | OPSD config | `4` | Must exceed `continuous_batch_size` to have pending requests |
| `rollout.max_prompt_length` | OPSD config | `2048` | Smaller first-run memory; raise for LongBench |
| `rollout.max_response_length` | OPSD config | `512` | Decode length; raise for longer ReTool traces |
| `training.max_steps` | OPSD config | `1` | One step; includes graph-capture cost |
| `training.logging_steps` | OPSD config | `1` | Per-step stdout |
| `wall_clock_breakdown` | DS JSON | `true` | DS fwd/bwd/step timers |
| `hybrid_engine.enabled` | DS JSON | `true` | Required for `DeepSpeedHybridEngine` |
| `hybrid_engine.enable_graph_capture` | DS JSON | `false` | This is the DS inference graph, not the rollout graph; Qwen3 has no DS inference containers |
| `zero_optimization.stage` | DS JSON | `1` | Keeps student params on GPU; avoids ZeRO-3 gather during graph capture |
| `zero_optimization.offload_optimizer` | DS JSON | CPU | Keeps 4B optimizer states off the GPU |
| `align_decode_fronts` | rollout config | `false` | Continuous graph rejects aligned fronts |
| `enable_cache_trimming` | rollout config | `false` | Continuous graph rejects trimming |

---

## 9. Generation-path matrix and constraints

The four generation paths:

| Path | Selection | What runs |
|---|---|---|
| Static eager | `continuous_batch_size=None`, `use_graph_capture=False` | HF `generate()` over the whole padded batch |
| Static graph | `continuous_batch_size=None`, `use_graph_capture=True` | Static decode forward captured with `DeepSpeedStaticCache` and replayed |
| Continuous eager | `continuous_batch_size=N`, `use_graph_capture=False` | Scheduler refills freed slots from pending requests |
| Continuous graph | `continuous_batch_size=N`, `use_graph_capture=True` | Fixed-capacity captured decode with continuous refill; **this baseline** |

Selection happens at `deepspeed/runtime/rollout/hybrid_engine_rollout.py:139-140`:

```python
if sampling.continuous_batch_size is not None:
    return self._generate_continuous(request, sampling, sampling.continuous_batch_size)
```

and the graph switch is `HybridEngineRolloutConfig.use_graph_capture`.

### Constraints enforced by the code

Continuous batching:

- Greedy only: `temperature <= 0`.
- `n_samples_per_prompt == 1`.
- One prompt row per request.
- All requests in the batch must have equal padded prompt width.
- `use_shared_prefill` must be false (it raises).
- The model must support cache classes (`_supports_cache_class`).
- The request must fit `max_position_embeddings`.

Continuous **graph** additionally:

- `align_decode_fronts` must be false.
- `enable_cache_trimming` must be false.
- `continuous_cache_capacity`, if set, must be at least `prompt_len + max_new_tokens`.
- `flash_attention_2` is rejected; use eager or SDPA.
- The accelerator must report graph-capture support.

### Graph capture cost and ZeRO caveat

- Graph capture happens on the first generation. With `max_steps=1`, the capture cost lands in your only measured step.
- The graph signature includes parameter `data_ptr()` and `_version`, so a weight update invalidates the graph and forces a recapture on the next step.
- The validated graph tests use ZeRO-0 / non-offloaded models. The baseline therefore uses **ZeRO-1 + optimizer offload** for the student, not ZeRO-3, so parameters stay on GPU and the forward is intended to be graph-capturable. Validate this on your Qwen3 run.
- The teacher still uses ZeRO-3 parameter offload; that is a separate engine and does not participate in the rollout graph.

### Metrics for the continuous paths

`HybridEngineRollout._last_profile` adds these fields on top of the static profile:

```text
num_prefill_forwards
num_decode_forwards
scheduler_overhead_ms
cache_management_overhead_ms
active_batch_size
continuous_batch_size
```

The prefill/decode/total/tokens-per-second fields are the same. `get_last_continuous_stats()` additionally reports cache capacity, peak cache length, cache memory bytes, trim count/latency, and decode steps.

The closest validated continuous+graph reference is:

```text
tests/unit/v1/test_hybrid_engine_rollout_v1.py::test_continuous_graph_generation_refills_a_fixed_slot_on_cuda
```

It uses a tiny `CacheClassModel`, not Qwen3, so it validates the mechanism, not the model path.

---

## 10. CPU-only ZeRO-3 `load_checkpoint` check

Not a GPU run; run it on the login node. It verifies `deepspeed.zero.Init`
plus `engine.load_checkpoint(..., load_module_strict=False)` on a CPU-only host
with the `gloo` backend.

```bash
source ~/opsd_env.sh
conda activate "$WORK/conda/opsd"
cd ~/DeepSpeedExamples/training/opsd

python - <<'PY'
import json, os, shutil, tempfile

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import deepspeed
from transformers import Qwen3Config, Qwen3ForCausalLM

deepspeed.init_distributed(dist_backend="gloo")

cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128)
ds_cfg = json.load(open("configs/ds_zero3.json"))
ds_cfg["train_micro_batch_size_per_gpu"] = 1
ds_cfg["scheduler"]["params"]["warmup_num_steps"] = 1
ds_cfg["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
ds_cfg["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
ds_cfg["hybrid_engine"] = {"enabled": False}

ckpt = tempfile.mkdtemp(prefix="zero3_ckpt_")
with deepspeed.zero.Init(config_dict_or_path=ds_cfg):
    model = Qwen3ForCausalLM(cfg)

engine, *_ = deepspeed.initialize(model=model, config=ds_cfg)
engine.save_checkpoint(ckpt, "step1")

engine2, *_ = deepspeed.initialize(model=model, config=ds_cfg)
path, _ = engine2.load_checkpoint(ckpt, "step1", load_module_strict=False)
print("load_checkpoint ok", path)
shutil.rmtree(ckpt, ignore_errors=True)
PY
```

Verified on this login node: `zero.Init on CPU ok`, `engine built on CPU`,
`save_checkpoint ok`, `load_checkpoint strict=False ok`.

Requirements:

- `gloo` backend: `deepspeed.init_distributed(dist_backend="gloo")`, or set
  `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `LOCAL_RANK`.
- `LOCAL_RANK` must be set; `zero.Init` reads `os.environ["LOCAL_RANK"]`.
- The DS config needs `train_micro_batch_size_per_gpu` (or `train_batch_size`).
- `WarmupLR` needs `warmup_num_steps >= 1`; `0` raises `ValueError`.
- Set `offload_param` and `offload_optimizer` to `cpu`; the CPU accelerator's
  `device_name()` returns `cpu`.
- Disable `hybrid_engine` for this check; it is a training-only concern and has
  no meaning on CPU.

---

## 11. Bottom line

- The training-step baseline is now continuous + graph: `use_graph_capture: true`, `continuous_batch_size: 2`, `micro_batch_size_per_gpu: 4`.
- The clean rollout-only baseline is `benchmarks/bench_continuous_graph.py`; it warms up once (captures the graph) and then reports steady-state replay iterations.
- For the training step, graph capture is part of every step because the optimizer update changes parameter versions. Read `rollout_profile` for prefill/decode/scheduler/cache breakdown, not only `rollout_s`.
- Qwen3 is still the native model path, not DeepSpeed's accelerated inference kernels. The continuous scheduler and `DeepSpeedStaticCache` are DeepSpeed's; the model forward is Qwen3's own.
