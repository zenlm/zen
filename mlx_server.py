#!/usr/bin/env python3
"""
Zen MLX Server - Native inference for foundational models on Apple Silicon
"""

import os
import json
import time
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import mlx
import mlx.core as mx
from mlx_lm import load, generate
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


# === Configuration ===
@dataclass
class ModelConfig:
    name: str
    path: str
    context_length: int
    quantization: str = "4bit"

MODELS = {
    "qwen3-4b-instruct": ModelConfig(
        name="Qwen3-4B-Instruct-2507",
        path="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        context_length=8192,
    ),
    "qwen3-4b-thinking": ModelConfig(
        name="Qwen3-4B-Thinking-2507",
        path="mlx-community/Qwen3-4B-Thinking-2507-4bit",
        context_length=8192,
    ),
    "qwen2.5-coder-3b": ModelConfig(
        name="Qwen2.5-Coder-3B-Instruct",
        path="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        context_length=32768,
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        path="mlx-community/Qwen2.5-7B-Instruct-4bit",
        context_length=32768,
    ),
}


# === Request/Response Models ===
class InferenceRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen3-4b-instruct"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "qwen3-4b-instruct"
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


class EmbeddingRequest(BaseModel):
    input: str
    model: Optional[str] = "qwen3-4b-instruct"


# === MLX Model Manager ===
class MLXModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None

    def load_model(self, model_name: str):
        """Load or switch to a model"""
        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not found")

        if model_name in self.models:
            self.current_model = model_name
            return

        print(f"Loading {model_name} from {MODELS[model_name].path}...")

        # Load model and tokenizer
        model, tokenizer = load(MODELS[model_name].path)

        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer
        self.current_model = model_name

        print(f"✓ Loaded {model_name}")

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.95) -> str:
        """Generate text using current model"""
        if not self.current_model:
            raise ValueError("No model loaded")

        model = self.models[self.current_model]
        tokenizer = self.tokenizers[self.current_model]

        # Generate response
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        )

        return response

    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings (using last hidden states)"""
        if not self.current_model:
            raise ValueError("No model loaded")

        model = self.models[self.current_model]
        tokenizer = self.tokenizers[self.current_model]

        # Tokenize
        tokens = tokenizer.encode(text)
        tokens = mx.array([tokens])

        # Get embeddings from model
        with mx.no_grad():
            # Forward pass through model
            outputs = model(tokens)
            # Use mean pooling of last hidden states
            embeddings = mx.mean(outputs, axis=1)
            embeddings = embeddings.squeeze(0)

        return embeddings.tolist()


# === FastAPI App ===
app = FastAPI(title="Zen MLX Server")
model_manager = MLXModelManager()


@app.on_event("startup")
async def startup():
    """Load default model on startup"""
    model_manager.load_model("qwen3-4b-instruct")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend": "MLX",
        "device": "Apple Silicon",
        "current_model": model_manager.current_model,
        "available_models": list(MODELS.keys()),
    }


@app.post("/v1/inference")
async def inference(request: InferenceRequest):
    """Simple inference endpoint"""
    try:
        # Switch model if needed
        if request.model != model_manager.current_model:
            model_manager.load_model(request.model)

        # Generate response
        response = model_manager.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return {
            "response": response,
            "model": request.model,
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(request.prompt.split()) + len(response.split()),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions"""
    try:
        # Switch model if needed
        if request.model != model_manager.current_model:
            model_manager.load_model(request.model)

        # Format messages into prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n"
        prompt += "Assistant: "

        # Generate response
        response = model_manager.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings"""
    try:
        # Switch model if needed
        if request.model != model_manager.current_model:
            model_manager.load_model(request.model)

        # Generate embeddings
        embedding = model_manager.get_embeddings(request.input)

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": 0
                }
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": len(request.input.split()),
                "total_tokens": len(request.input.split()),
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = []
    for key, config in MODELS.items():
        models.append({
            "id": key,
            "name": config.name,
            "context_length": config.context_length,
            "quantization": config.quantization,
            "loaded": key in model_manager.models,
        })

    return {"models": models}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zen MLX Server")
    parser.add_argument("--model", default="qwen3-4b-instruct", help="Initial model to load")
    parser.add_argument("--port", type=int, default=3690, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")

    args = parser.parse_args()

    # Load initial model
    print(f"Starting Zen MLX Server on port {args.port}")
    print(f"Loading model: {args.model}")

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)