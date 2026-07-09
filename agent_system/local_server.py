"""Minimal OpenAI-compatible local server for Qwen3-VL.  ~100 lines, no heavy deps."""
import json, os, time, base64, io
from typing import Optional
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/media/data6/xuejj/Qwen")
model = None
processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("Model loaded.")
    yield


app = FastAPI(lifespan=lifespan)


def _build_messages(raw_msgs: list) -> list:
    """Convert OpenAI-format messages (with base64 images) to Qwen format."""
    out = []
    for m in raw_msgs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        # content is [{"type":"text","text":"..."}, {"type":"image_url",...}, ...]
        items = []
        for part in content:
            if part.get("type") == "text":
                items.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    b64_data = url.split(",", 1)[1]
                else:
                    b64_data = url
                items.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_data}"})
        out.append({"role": role, "content": items})
    return out


def _extract_images(raw_msgs: list) -> list:
    images = []
    for m in raw_msgs:
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:image/"):
                url = url.split(",", 1)[1]
            images.append(Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB"))
    return images


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    raw_msgs = body.get("messages", [])
    msgs = _build_messages(raw_msgs)
    images = _extract_images(raw_msgs)
    max_tokens = min(body.get("max_tokens", 1024), 4096)
    temperature = body.get("temperature", 0.2)

    # Apply chat template
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

    # Decode only the new tokens
    input_len = inputs.input_ids.shape[1]
    new_tokens = generated[0][input_len:]
    response_text = processor.decode(new_tokens, skip_special_tokens=True)

    resp_id = f"local_{int(time.time()*1000)}"
    return JSONResponse({
        "id": resp_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": os.path.basename(MODEL_PATH),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
    })


if __name__ == "__main__":
    port = int(os.environ.get("LOCAL_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
