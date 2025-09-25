#!/usr/bin/env python3.11
"""
Zen1 MLX Inference Server
Provides OpenAI-compatible API for fine-tuned Qwen3-4B models
"""

import os
import time
import json
import uuid
import asyncio
from typing import Optional, Dict, List, AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

import mlx
import mlx.core as mx
from mlx_lm import load, generate

# Configuration
BASE_MODEL_INSTRUCT = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
BASE_MODEL_THINKING = "mlx-community/Qwen3-4B-Thinking-2507-4bit"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7

# Global model storage
models = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    stream: Optional[bool] = False
    show_thinking: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("Loading Zen1 models...")

    # Load instruction model
    print(f"Loading instruction model: {BASE_MODEL_INSTRUCT}")
    models["zen1-instruct"], models["tokenizer-instruct"] = load(BASE_MODEL_INSTRUCT)
    print("✓ Instruction model loaded")

    # Optionally load thinking model
    # print(f"Loading thinking model: {BASE_MODEL_THINKING}")
    # models["zen1-thinking"], models["tokenizer-thinking"] = load(BASE_MODEL_THINKING)
    # print("✓ Thinking model loaded")

    yield

    # Cleanup
    models.clear()
    print("Models unloaded")


app = FastAPI(title="Zen1 MLX Server", version="1.0.0", lifespan=lifespan)


def format_messages(messages: List[ChatMessage], variant: str = "instruct") -> str:
    """Format messages for the model"""
    if variant == "thinking":
        # Format for chain-of-thought
        formatted = ""
        for msg in messages:
            if msg.role == "user":
                formatted += f"Question: {msg.content}\\n\\n<think>\\n"
            elif msg.role == "assistant":
                formatted += f"{msg.content}\\n\\n"
        return formatted
    else:
        # Standard instruction format
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted = f"System: {msg.content}\\n\\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\\n\\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\\n\\n"
        formatted += "Assistant:"
        return formatted


def extract_answer(response: str, variant: str = "instruct") -> str:
    """Extract answer from response"""
    if variant == "thinking" and "Answer:" in response:
        parts = response.split("Answer:")
        return parts[-1].strip()
    return response.strip()


@app.get("/")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": list(models.keys()),
        "version": "1.0.0"
    }


@app.get("/v1/models")
async def list_models():
    """List available models"""
    available_models = []

    if "zen1-instruct" in models:
        available_models.append({
            "id": "zen1-instruct",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zen"
        })

    if "zen1-thinking" in models:
        available_models.append({
            "id": "zen1-thinking",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zen"
        })

    return {"data": available_models, "object": "list"}


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint"""

    # Determine model variant
    if "thinking" in request.model.lower():
        variant = "thinking"
        model_key = "zen1-thinking"
        tokenizer_key = "tokenizer-thinking"
    else:
        variant = "instruct"
        model_key = "zen1-instruct"
        tokenizer_key = "tokenizer-instruct"

    # Check if model is loaded
    if model_key not in models:
        # Fallback to instruct if thinking not loaded
        if variant == "thinking" and "zen1-instruct" in models:
            variant = "instruct"
            model_key = "zen1-instruct"
            tokenizer_key = "tokenizer-instruct"
        else:
            raise HTTPException(status_code=404, detail=f"Model {request.model} not found")

    # Get model and tokenizer
    model = models[model_key]
    tokenizer = models[tokenizer_key]

    # Format prompt
    prompt = format_messages(request.messages, variant)

    # Generate response
    if request.stream:
        return StreamingResponse(
            stream_response(model, tokenizer, prompt, request),
            media_type="text/event-stream"
        )

    # Non-streaming response
    response_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        verbose=False
    )

    # Extract answer if needed
    if variant == "thinking" and not request.show_thinking:
        response_text = extract_answer(response_text, variant)

    # Format response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(response_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
        }
    )

    return response


async def stream_response(model, tokenizer, prompt: str, request: ChatCompletionRequest) -> AsyncIterator[str]:
    """Stream response chunks"""
    # Generate complete response (MLX doesn't support true streaming yet)
    response_text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=request.max_tokens,
        verbose=False
    )

    # Simulate streaming
    chunk_size = 20  # Characters per chunk
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    for i in range(0, len(response_text), chunk_size):
        chunk_text = response_text[i:i+chunk_size]

        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk_text},
                "finish_reason": None
            }]
        }

        yield f"data: {json.dumps(chunk)}\\n\\n"

        # Small delay to simulate streaming
        await asyncio.sleep(0.01)

    # Send finish chunk
    finish_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }

    yield f"data: {json.dumps(finish_chunk)}\\n\\n"
    yield "data: [DONE]\\n\\n"


@app.post("/v1/completions")
async def completion(request: Dict):
    """OpenAI-compatible completion endpoint (legacy)"""
    # Convert to chat format
    chat_request = ChatCompletionRequest(
        model=request.get("model", "zen1-instruct"),
        messages=[ChatMessage(role="user", content=request["prompt"])],
        max_tokens=request.get("max_tokens", DEFAULT_MAX_TOKENS),
        temperature=request.get("temperature", DEFAULT_TEMPERATURE),
        stream=request.get("stream", False)
    )

    return await chat_completion(chat_request)


if __name__ == "__main__":
    import uvicorn
    import asyncio

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting Zen1 MLX Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)