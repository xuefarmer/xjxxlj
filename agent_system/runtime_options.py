import argparse
import os
from typing import Dict, Optional


def parse_runtime_args(description: Optional[str] = None):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--master-backend", choices=["qwen3", "local_hf", "local", "hf", "api"],
                        help="Master reasoning backend. qwen3/local_hf loads a local HuggingFace Qwen3 model; api uses MASTER_API_*.")
    parser.add_argument("--master-model-path", help="Local Qwen3/HuggingFace master model path.")
    parser.add_argument("--master-api-base-url", help="OpenAI-compatible master API /v1/chat/completions URL.")
    parser.add_argument("--master-api-key", help="Master API key.")
    parser.add_argument("--master-model-name", help="Master API model name.")
    parser.add_argument("--master-max-tokens", type=int, help="Master API max tokens.")
    parser.add_argument("--local-master-max-new-tokens", type=int, help="Local master max new tokens.")
    parser.add_argument("--local-master-device", help="Local master device, e.g. cuda:0, cuda:1, cpu.")
    parser.add_argument("--local-master-device-map", help="Local master device_map, e.g. auto, cuda:0.")
    parser.add_argument("--local-master-dtype", help="Local master torch dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--local-master-enable-thinking", action="store_true",
                        help="Enable Qwen thinking mode in chat template for local master.")

    parser.add_argument("--tool-backend", choices=["qwenvl", "local_qwenvl", "local_vlm", "api"],
                        help="Visual observation backend. qwenvl/local_qwenvl loads local QwenVL; api uses TOOL_API_*.")
    parser.add_argument("--tool-model-path", help="Local QwenVL/HuggingFace model path.")
    parser.add_argument("--tool-api-base-url", help="OpenAI-compatible visual API /v1/chat/completions URL.")
    parser.add_argument("--tool-api-key", help="Visual API key.")
    parser.add_argument("--tool-model-name", help="Visual API model name.")
    parser.add_argument("--tool-max-tokens", type=int, help="Visual API max tokens.")
    parser.add_argument("--local-tool-max-new-tokens", type=int, help="Local QwenVL max new tokens.")
    parser.add_argument("--local-tool-device", help="Local QwenVL device, e.g. cuda:0, cuda:1, cpu.")
    parser.add_argument("--local-tool-device-map", help="Local QwenVL device_map, e.g. auto, cuda:0.")
    parser.add_argument("--local-tool-dtype", help="Local QwenVL torch dtype, e.g. auto, bfloat16, float16.")

    parser.add_argument("--prompt-dir",
                        help="Prompt directory. Relative paths are resolved under agent_system/.")
    parser.add_argument("--prompt-file",
                        help="Exact master prompt file. Relative paths are resolved under agent_system/.")
    parser.add_argument("--start", type=int, help="Start question number, 1-based.")
    parser.add_argument("--end", type=int, help="End question number, inclusive.")
    parser.add_argument("--max-turns", type=int, help="Agent max turns.")
    parser.add_argument("--oracle-answer-hint", action="store_true",
                        help="Debug only: pass the ground-truth answer to the master so it can generate oracle-guided diagnostic logs.")

    return parser.parse_args()


def _set_env(name: str, value):
    if value is not None:
        os.environ[name] = str(value)


def _normalize_master_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value in {"qwen3", "local", "hf"}:
        return "local_hf"
    return value


def _normalize_tool_backend(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value in {"qwenvl", "local_vlm"}:
        return "local_qwenvl"
    return value


def apply_runtime_overrides(args) -> None:
    _set_env("MASTER_BACKEND", _normalize_master_backend(args.master_backend))
    _set_env("LOCAL_MASTER_MODEL_PATH", args.master_model_path)
    _set_env("MASTER_API_BASE_URL", args.master_api_base_url)
    _set_env("MASTER_API_KEY", args.master_api_key)
    _set_env("MASTER_MODEL_NAME", args.master_model_name)
    _set_env("MASTER_MAX_TOKENS", args.master_max_tokens)
    _set_env("LOCAL_MASTER_MAX_NEW_TOKENS", args.local_master_max_new_tokens)
    _set_env("LOCAL_MASTER_DEVICE", args.local_master_device)
    _set_env("LOCAL_MASTER_DEVICE_MAP", args.local_master_device_map)
    _set_env("LOCAL_MASTER_TORCH_DTYPE", args.local_master_dtype)
    if args.local_master_enable_thinking:
        os.environ["LOCAL_MASTER_ENABLE_THINKING"] = "true"
    if args.oracle_answer_hint:
        os.environ["AGENT_ORACLE_ANSWER_HINT"] = "true"

    _set_env("TOOL_BACKEND", _normalize_tool_backend(args.tool_backend))
    _set_env("LOCAL_TOOL_MODEL_PATH", args.tool_model_path)
    _set_env("TOOL_API_BASE_URL", args.tool_api_base_url)
    _set_env("TOOL_API_KEY", args.tool_api_key)
    _set_env("TOOL_MODEL_NAME", args.tool_model_name)
    _set_env("TOOL_MAX_TOKENS", args.tool_max_tokens)
    _set_env("LOCAL_TOOL_MAX_NEW_TOKENS", args.local_tool_max_new_tokens)
    _set_env("LOCAL_TOOL_DEVICE", args.local_tool_device)
    _set_env("LOCAL_TOOL_DEVICE_MAP", args.local_tool_device_map)
    _set_env("LOCAL_TOOL_TORCH_DTYPE", args.local_tool_dtype)


def _resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def resolve_prompt_config(root: str, task_code: str, default_config: Dict[str, str], args) -> Dict[str, str]:
    config = dict(default_config)
    prompt_file = args.prompt_file or os.environ.get("AGENT_PROMPT_FILE", "").strip()
    prompt_dir = args.prompt_dir or os.environ.get("AGENT_PROMPT_DIR", "").strip()

    if prompt_file:
        config["master"] = _resolve_path(root, prompt_file)
    elif prompt_dir:
        prompt_dir_path = _resolve_path(root, prompt_dir)
        unified_base = os.path.join(prompt_dir_path, "_unified_crossvid_base.prompt")
        task_contract = os.path.join(prompt_dir_path, "task_contracts", f"{task_code}.prompt")
        if os.path.exists(unified_base) and os.path.exists(task_contract):
            config["master_base"] = unified_base
            config["task_contract"] = task_contract
            config["task_code"] = task_code
            # Keep a logical name for executor compatibility checks and logs.
            config["master_name"] = f"master_{task_code}.prompt"
            config.pop("master", None)
        else:
            config["master"] = os.path.join(prompt_dir_path, f"master_{task_code}.prompt")

    return config


def runtime_summary(prompt_config: Dict[str, str]) -> str:
    prompt_desc = prompt_config.get("master")
    if not prompt_desc and prompt_config.get("master_base") and prompt_config.get("task_contract"):
        prompt_desc = f"{prompt_config.get('master_base')} + {prompt_config.get('task_contract')}"
    return (
        f"MASTER_BACKEND={os.environ.get('MASTER_BACKEND', 'local_hf')} | "
        f"LOCAL_MASTER_MODEL_PATH={os.environ.get('LOCAL_MASTER_MODEL_PATH', '/media/data6/xuejj/Qwen3-8B')} | "
        f"MASTER_MODEL_NAME={os.environ.get('MASTER_MODEL_NAME', '')} | "
        f"TOOL_BACKEND={os.environ.get('TOOL_BACKEND', 'api')} | "
        f"LOCAL_TOOL_MODEL_PATH={os.environ.get('LOCAL_TOOL_MODEL_PATH', '/media/data6/xuejj/Qwen')} | "
        f"TOOL_MODEL_NAME={os.environ.get('TOOL_MODEL_NAME', '')} | "
        f"ORACLE_ANSWER_HINT={os.environ.get('AGENT_ORACLE_ANSWER_HINT', 'false')} | "
        f"PROMPT={prompt_desc}"
    )
