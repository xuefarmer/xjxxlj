# CCQA/PEA staged local evaluation

The code repository does not contain CrossVid videos/questions, base models,
LoRA adapters, preheat reports, or evaluation outputs. Supply those paths on
the compute node. Run from the repository root in the environment with
PyTorch, Transformers, PEFT, OpenCV, and the dependencies in `pyproject.toml`.

Preheat Event Observer reports. Each physical GPU gets one VLM process and a
deterministic subset of rows; rerunning skips valid cached reports:

```bash
python scripts/preheat_event_local_vlm.py \
  --tasks CCQA,PEA --gpus 0,1 \
  --config configs/local_dual_lora_eval_longvideo_compact.yaml \
  --cache-dir outputs/eval-event-rl200-staged/preheat-ccqa-pea \
  --vlm-model /path/to/Qwen-VL \
  --vlm-adapter /path/to/step-000200/vlm \
  --crossvid-root /path/to/CrossVid
```

Confirm that the cache contains 872 CCQA and 953 PEA JSON reports before
evaluation. Rerun preheat with the same arguments if either count is short.

Evaluate on one physical GPU. The LLM and VLM are both resident on the one
card but generate in alternating calls. Use a new output directory for each
experimental version; the wrapper resumes completed sample IDs on rerun:

```bash
python scripts/eval_cached_single_gpu.py \
  --physical-gpu 0 --tasks CCQA,PEA \
  --config configs/local_dual_lora_eval_longvideo_compact.yaml \
  --cache-dir outputs/eval-event-rl200-staged/preheat-ccqa-pea \
  --output-dir outputs/eval-event-rl200-staged/ccqa-pea-rl200-single-a100 \
  --llm-model /path/to/Qwen3-8B \
  --llm-adapter /path/to/step-000200/llm \
  --vlm-model /path/to/Qwen-VL \
  --vlm-adapter /path/to/step-000200/vlm \
  --crossvid-root /path/to/CrossVid
```

CCQA's built-in score is a deterministic scoring-point coverage *proxy*, not
the benchmark's external model-judge score. PEA uses exact-match scoring.
