import abc
import json
import os
import sys
import time
import requests
from typing import List, Any, Dict, Optional, Union
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

device = "cpu"


class BaseModel(abc.ABC):
    requires_gpu = False
    def __init__(self, gpu_number: int = 0): self.dev = device
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str: ...
    @classmethod
    def list_processes(cls) -> List[str]: return [cls.name()]

class QwenModel(BaseModel):
    _name = 'qwen_layered_agent'
    requires_gpu = False

    def __init__(self, gpu_number=0, prompt_config: Optional[Dict[str, str]] = None):
        super().__init__(gpu_number)
        self.master_backend = os.environ.get("MASTER_BACKEND", "local_hf").strip().lower()
        self.local_master_config = {
            "model_path": os.environ.get("LOCAL_MASTER_MODEL_PATH", "/media/data6/xuejj/Qwen3-8B"),
            "max_new_tokens": int(os.environ.get("LOCAL_MASTER_MAX_NEW_TOKENS", os.environ.get("MASTER_MAX_TOKENS", "8192"))),
            "temperature": float(os.environ.get("LOCAL_MASTER_TEMPERATURE", os.environ.get("MASTER_TEMPERATURE", "0.2"))),
            "top_p": float(os.environ.get("LOCAL_MASTER_TOP_P", "0.95")),
            "top_k": int(os.environ.get("LOCAL_MASTER_TOP_K", "20")),
            "device": os.environ.get("LOCAL_MASTER_DEVICE", "").strip(),
            "device_map": os.environ.get("LOCAL_MASTER_DEVICE_MAP", "auto"),
            "torch_dtype": os.environ.get("LOCAL_MASTER_TORCH_DTYPE", "auto"),
            "enable_thinking": os.environ.get("LOCAL_MASTER_ENABLE_THINKING", "false").strip().lower() in {"1", "true", "yes", "on"},
        }
        self._local_master_tokenizer = None
        self._local_master_model = None
        self._local_master_pipeline = None
        self.tool_backend = os.environ.get("TOOL_BACKEND", "api").strip().lower()
        self.local_tool_config = {
            "model_path": os.environ.get("LOCAL_TOOL_MODEL_PATH", os.environ.get("LOCAL_VLM_MODEL_PATH", "/media/data6/xuejj/Qwen")),
            "max_new_tokens": int(os.environ.get("LOCAL_TOOL_MAX_NEW_TOKENS", os.environ.get("TOOL_MAX_TOKENS", "8192"))),
            "temperature": float(os.environ.get("LOCAL_TOOL_TEMPERATURE", os.environ.get("TOOL_TEMPERATURE", "0.2"))),
            "top_p": float(os.environ.get("LOCAL_TOOL_TOP_P", "0.95")),
            "top_k": int(os.environ.get("LOCAL_TOOL_TOP_K", "20")),
            "device": os.environ.get("LOCAL_TOOL_DEVICE", "").strip(),
            "device_map": os.environ.get("LOCAL_TOOL_DEVICE_MAP", "auto"),
            "torch_dtype": os.environ.get("LOCAL_TOOL_TORCH_DTYPE", "auto"),
        }
        self._local_tool_processor = None
        self._local_tool_model = None

        # API config from env (no secrets in repo). MASTER_API_* is kept as a fallback
        # and as the default source for the diagnosis agent API.
        self.master_config = {
            "base_url": os.environ.get("MASTER_API_BASE_URL", "http://example-master-api.example.com/v1/chat/completions"),
            "api_key": os.environ.get("MASTER_API_KEY", "your-api-key"),
            "model_name": os.environ.get("MASTER_MODEL_NAME", "your-master-model"),
            "max_tokens": int(os.environ.get("MASTER_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("MASTER_TEMPERATURE", "0.5")),
        }

        self.diagnosis_config = {
            "base_url": os.environ.get("DIAGNOSIS_API_BASE_URL", self.master_config["base_url"]),
            "api_key": os.environ.get("DIAGNOSIS_API_KEY", self.master_config["api_key"]),
            "model_name": os.environ.get("DIAGNOSIS_MODEL_NAME", self.master_config["model_name"]),
            "max_tokens": int(os.environ.get("DIAGNOSIS_MAX_TOKENS", str(self.master_config["max_tokens"]))),
            "temperature": float(os.environ.get("DIAGNOSIS_TEMPERATURE", str(self.master_config["temperature"]))),
        }

        self.tool_config = {
            "base_url": os.environ.get("TOOL_API_BASE_URL", "http://example-tool-api.example.com/v1/chat/completions"),
            "api_key": os.environ.get("TOOL_API_KEY", "your-api-key"),
            "model_name": os.environ.get("TOOL_MODEL_NAME", "your-tool-model"),
            "max_tokens": int(os.environ.get("TOOL_MAX_TOKENS", "8192")),
            "temperature": float(os.environ.get("TOOL_TEMPERATURE", "0.5")),
        }

        self.prompt_config = dict(prompt_config) if prompt_config else {}
        self.last_tool_prompt: Optional[str] = None
        self.last_tool_name: Optional[str] = None
        self.last_tool_images: int = 0

    @classmethod
    def name(cls): return cls._name

    def forward(self, task: str, **kwargs) -> Any:
        """Dispatches to master (chat) or active_perception (VLM); matches executor tools."""
        router = {
            'master': self._master,
            'active_perception': self._active_perception,
            'diagnosis': self._diagnosis,
        }
        handler = router.get(task)
        if not handler:
            raise ValueError(f"Unknown task '{task}'. Supported: master, active_perception, diagnosis.")
        return handler(**kwargs)

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        text_parts.append(str(item.get("text", "")))
                    else:
                        text_parts.append(str(item))
                else:
                    text_parts.append(str(item))
            return "\n".join(part for part in text_parts if part)
        if content is None:
            return ""
        return str(content)

    @classmethod
    def _downgrade_tool_messages(cls, messages: list) -> list:
        out = []
        for msg in messages or []:
            role = msg.get("role", "user")
            text = cls._message_content_to_text(msg.get("content", ""))
            if role == "tool":
                out.append({"role": "user", "content": f"[TOOL OBS] {text}"})
            else:
                out.append({"role": role, "content": text})
        return out

    def _load_local_master(self):
        if self._local_master_pipeline is not None and self._local_master_tokenizer is not None:
            return self._local_master_tokenizer, self._local_master_pipeline

        if sys.version_info < (3, 9):
            raise RuntimeError(
                "Local Qwen3-8B master inference needs a newer Python environment. "
                f"Current Python is {sys.version.split()[0]}; use Python >= 3.9, preferably 3.10/3.11."
            )

        try:
            import torch
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError(
                "Local MASTER_BACKEND=local_hf uses transformers.pipeline and requires torch + transformers. "
                "Install them in the same Python environment that runs run_PSS_agent.py."
            ) from exc

        model_path = self.local_master_config["model_path"]
        dtype_name = self.local_master_config["torch_dtype"]
        if dtype_name == "auto":
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype_name)

        explicit_device = self.local_master_config["device"]
        device_map = self.local_master_config["device_map"]
        if explicit_device:
            device_map = {"": explicit_device}
        elif isinstance(device_map, str):
            device_map = device_map.strip()
            if device_map.isdigit():
                device_map = {"": f"cuda:{device_map}"}
            elif device_map.startswith(("cuda", "cpu", "mps")):
                device_map = {"": device_map}

        logger.info("  [QwenModel] Loading local MASTER model with transformers.pipeline from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text_generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        self._local_master_tokenizer = tokenizer
        self._local_master_pipeline = text_generator
        self._local_master_model = getattr(text_generator, "model", None)
        if self._local_master_model is not None:
            self._local_master_model.eval()
        return tokenizer, text_generator

    def _call_local_master_chat(self, messages: list) -> str:
        tokenizer, text_generator = self._load_local_master()
        norm_messages = self._downgrade_tool_messages(messages)

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt_text = tokenizer.apply_chat_template(
                norm_messages,
                enable_thinking=self.local_master_config["enable_thinking"],
                **template_kwargs,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(norm_messages, **template_kwargs)

        temperature = self.local_master_config["temperature"]
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.local_master_config["max_new_tokens"],
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
            "return_full_text": False,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": self.local_master_config["top_p"],
                "top_k": self.local_master_config["top_k"],
            })

        outputs = text_generator(prompt_text, **generation_kwargs)
        if not outputs:
            return ""
        generated_text = outputs[0].get("generated_text", "") if isinstance(outputs[0], dict) else str(outputs[0])
        if generated_text.startswith(prompt_text):
            generated_text = generated_text[len(prompt_text):]
        return generated_text.strip()

    @staticmethod
    def _resolve_torch_dtype(torch_module, dtype_name: str):
        if dtype_name == "auto":
            return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32
        if not hasattr(torch_module, dtype_name):
            raise ValueError(f"Unknown torch dtype '{dtype_name}'. Try auto, bfloat16, float16, or float32.")
        return getattr(torch_module, dtype_name)

    @staticmethod
    def _resolve_device_map(device: str, device_map: str):
        if device:
            return {"": device}
        if isinstance(device_map, str):
            device_map = device_map.strip()
            if device_map.isdigit():
                return {"": f"cuda:{device_map}"}
            if device_map.startswith(("cuda", "cpu", "mps")):
                return {"": device_map}
        return device_map

    def _load_local_tool(self):
        if self._local_tool_processor is not None and self._local_tool_model is not None:
            return self._local_tool_processor, self._local_tool_model

        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Local TOOL_BACKEND=local_qwenvl requires torch + transformers. "
                "Install agent_system/requirements.txt in the Python environment that runs the task."
            ) from exc

        model_path = self.local_tool_config["model_path"]
        torch_dtype = self._resolve_torch_dtype(torch, self.local_tool_config["torch_dtype"])
        device_map = self._resolve_device_map(
            self.local_tool_config["device"],
            self.local_tool_config["device_map"],
        )

        model_cls = None
        tried = []
        for class_name in (
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "AutoModelForImageTextToText",
        ):
            candidate = getattr(transformers, class_name, None)
            if candidate is not None:
                model_cls = candidate
                break
            tried.append(class_name)

        if model_cls is None:
            raise ImportError(
                "Your transformers build does not expose a Qwen VL model class. "
                f"Tried: {', '.join(tried)}. Install a transformers version that supports Qwen3-VL/Qwen2.5-VL."
            )

        logger.info("  [QwenModel] Loading local TOOL QwenVL from %s with %s", model_path, model_cls.__name__)
        model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model.eval()

        self._local_tool_processor = processor
        self._local_tool_model = model
        return processor, model

    @staticmethod
    def _decode_base64_images(base64_images: Optional[List[str]]) -> List[Any]:
        if not base64_images:
            return []
        import base64
        import io
        from PIL import Image

        images = []
        for img_b64 in base64_images:
            if "," in img_b64 and img_b64.strip().startswith("data:image/"):
                img_b64 = img_b64.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            images.append(image)
        return images

    @staticmethod
    def _interleave_text_and_images(prompt_text: str, images: List[Any]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if images and "<frame>" in prompt_text:
            prompt_splits = prompt_text.split("<frame>")
            for idx, split in enumerate(prompt_splits):
                if split:
                    content.append({"type": "text", "text": split})
                if idx < len(images):
                    content.append({"type": "image", "image": images[idx]})
            return content

        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})
        return content

    @staticmethod
    def _move_inputs_to_model_device(inputs: Any, model: Any):
        target_device = getattr(model, "device", None)
        if target_device is None:
            try:
                target_device = next(model.parameters()).device
            except Exception:
                target_device = None
        if target_device is not None and hasattr(inputs, "to"):
            return inputs.to(target_device)
        return inputs

    def _call_local_tool_vlm(self, prompt_text: str, base64_images: Optional[List[str]] = None) -> str:
        processor, model = self._load_local_tool()
        images = self._decode_base64_images(base64_images)
        messages = [{
            "role": "user",
            "content": self._interleave_text_and_images(prompt_text, images),
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images or None, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs, model)

        temperature = self.local_tool_config["temperature"]
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.local_tool_config["max_new_tokens"],
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": self.local_tool_config["top_p"],
                "top_k": self.local_tool_config["top_k"],
            })

        try:
            import torch
            context = torch.inference_mode()
        except Exception:
            context = None

        if context is None:
            generated_ids = model.generate(**inputs, **generation_kwargs)
        else:
            with context:
                generated_ids = model.generate(**inputs, **generation_kwargs)

        input_len = inputs.input_ids.shape[1]
        generated_trimmed = generated_ids[:, input_len:]
        decoded = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""

    def _call_api(self, prompt_text: str, base64_images: Optional[List[str]] = None, expect_json: bool = False, config: Dict = None) -> str:
        api_cfg = config or self.tool_config

        headers = {"Authorization": f"Bearer {api_cfg['api_key']}", "Content-Type": "application/json"}
        logger.debug(f"  [_call_api] Using Model: {api_cfg['model_name']} @ {api_cfg['base_url']}")

        content_list = []
        if base64_images and "<frame>" in prompt_text:
            prompt_splits = prompt_text.split('<frame>')
            expected_placeholders = len(prompt_splits) - 1
            if expected_placeholders != len(base64_images):
                logger.warning(
                    f"  [_call_api] Mismatch! Prompt has {expected_placeholders} <frame> tags, "
                    f"but {len(base64_images)} images were provided. "
                    f"Will iterate up to {min(expected_placeholders, len(base64_images))} images."
                )

            for idx, split in enumerate(prompt_splits):
                if split:
                    content_list.append({"type": "text", "text": split})
                if idx < len(base64_images):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_images[idx]}"}
                    })

        else:
            if "<frame>" in prompt_text:
                logger.warning("  [_call_api] <frame> tags in prompt but no images provided.")
            if base64_images:
                for img_b64 in base64_images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
            content_list.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_list}]
        payload: Dict[str, Any] = {
            "model": api_cfg['model_name'],
            "messages": messages,
            "max_tokens": api_cfg['max_tokens'],
            "temperature": api_cfg['temperature']
        }
        if expect_json:
            payload['result_format'] = 'message'

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"  [_call_api] Retry {attempt + 1}/{max_retries}...")
                response = requests.post(api_cfg['base_url'], headers=headers, json=payload, timeout=1200)
                response.raise_for_status()

                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                if content:
                    return content
                else:
                    logger.warning(f"  [_call_api] API returned empty content (Attempt {attempt + 1}).")
                    if attempt < max_retries - 1:
                        raise ValueError("Empty content received")
                    return "[API returned empty content]"

            except Exception as e:
                logger.error(f"  [_call_api] Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"Error: {e}"


    def _master(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        if self.master_backend in {"local_hf", "local", "hf"}:
            logger.info("  [QwenModel] 🧠 Routing to LOCAL MASTER Qwen3-8B")
            if messages is not None:
                return self._call_local_master_chat(messages)
            return self._call_local_master_chat([{"role": "user", "content": prompt or ""}])

        logger.info("  [QwenModel] 🧠 Routing to MASTER Agent API")
        if messages is not None:
            return self._call_api_chat(messages=messages, expect_json=True, config=self.master_config)
        return self._call_api(prompt_text=prompt, expect_json=True, config=self.master_config)

    def _diagnosis(self, prompt: str = None, messages: list = None, **kwargs) -> str:
        logger.info("  [QwenModel] 🩺 Routing to DIAGNOSIS Agent API")
        if messages is not None:
            return self._call_api_chat(messages=messages, expect_json=True, config=self.diagnosis_config)
        return self._call_api(prompt_text=prompt, expect_json=True, config=self.diagnosis_config)

    def _active_perception(self, frames: List[str], prompt: str, return_prompt: bool = False) -> Union[str, Dict[str, Any]]:
        frame_markers = "".join(f"<frame>" for _ in range(len(frames or [])))
        prompt_with_markers = prompt + "\n" + frame_markers

        self.last_tool_prompt = prompt_with_markers
        self.last_tool_name = "active_perception"
        self.last_tool_images = len(frames or [])
        if self.tool_backend in {"local_qwenvl", "qwenvl", "local_vlm"}:
            logger.info("  [QwenModel] 👁️ Routing to LOCAL TOOL QwenVL (Active Perception)")
            result = self._call_local_tool_vlm(prompt_text=prompt_with_markers, base64_images=frames)
        else:
            logger.info("  [QwenModel] 👁️ Routing to TOOL Agent API (Active Perception)")
            result = self._call_api(prompt_text=prompt_with_markers, base64_images=frames, config=self.tool_config)
        return {"result": result, "prompt": prompt_with_markers} if return_prompt else result

    def _call_api_chat(self, messages: list, expect_json: bool = False, config: Dict = None) -> str:
        api_cfg = config or self.master_config

        headers = {
            "Authorization": f"Bearer {api_cfg['api_key']}",
            "Content-Type": "application/json"
        }
        base_url = api_cfg["base_url"]
        is_openai_like = "/v1/chat/completions" in base_url or "compatible-mode" in base_url
        force_downgrade = api_cfg.get("force_user_role", True)

        payload: Dict[str, Any] = {}
        if is_openai_like:
            msgs_for_send = self._downgrade_tool_messages(messages) if force_downgrade else messages
            payload = {
                "model": api_cfg["model_name"],
                "messages": msgs_for_send,
                "max_tokens": api_cfg.get("max_tokens", 1024),
                "temperature": api_cfg.get("temperature", 0.2),
            }
        else:
            norm_msgs = self._downgrade_tool_messages(messages)
            payload = {
                "model": api_cfg["model_name"],
                "input": {"messages": norm_msgs},
                "parameters": {}
            }
            if expect_json: payload["parameters"]["result_format"] = "message"

        # Retry Loop
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(base_url, headers=headers, json=payload, timeout=1200)
                resp.raise_for_status()
                data = resp.json()
                content = (((data.get("choices") or [{}])[0].get("message") or {}) or {}).get("content")
                if not content: content = data.get("output_text") or data.get("content")

                if not content or not str(content).strip(): raise ValueError("Empty content")
                return content.strip()
            except Exception as e:
                logger.error(f"🟥 [QWEN CHAT] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1: time.sleep(3); continue
                return json.dumps({"type": "error", "content": f"Failed: {e}"}, ensure_ascii=False)
