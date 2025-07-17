import os
import sys
import io
import json
import time
import datetime
import tempfile
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the core logic from traegpt.py
from traegpt import (
    ImageRecognitionSystem,
    SYSTEM_PROMPT,
    MODEL,
    OLLAMA_URL
)
import requests

# ========== FASTAPI SETUP ==========
app = FastAPI(title="TraeGPT API", description="Chat + Image Recognition API", version="1.0")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== MODELS ==========
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]

# ========== CHAT ENDPOINT (OpenAI-compatible) ==========
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatRequest):
    # Compose prompt from messages
    prompt = SYSTEM_PROMPT
    for msg in request.messages:
        if msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"AI: {msg.content}\n"
    prompt += "AI:"
    # Call Ollama API (streaming off for now)
    ollama_payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    ai_response = data.get("response", "")
    # Format OpenAI-style response
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=MODEL,
        choices=[ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=ai_response),
            finish_reason="stop"
        )]
    )

# ========== IMAGE ANALYSIS ENDPOINT ==========
@app.post("/v1/image/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    device: Optional[str] = Form(None),
    languages: Optional[str] = Form('en'),
    categories: Optional[str] = Form(None),
    analysis_type: Optional[str] = Form('full')
):
    # Save uploaded file to a temp location using tempfile
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Parse options
        ocr_languages = [lang.strip() for lang in languages.split(',')]
        custom_categories = [cat.strip() for cat in categories.split(',')] if categories else None
        system = ImageRecognitionSystem(device=device, ocr_languages=ocr_languages, custom_categories=custom_categories)
        # Run analysis
        if analysis_type == 'full':
            results = system.analyze_image(temp_path)
        elif analysis_type == 'classification':
            results = {"image_path": temp_path, "classification": system.classify_image(temp_path), "analysis_time": 0}
        elif analysis_type == 'detection':
            results = {"image_path": temp_path, "object_detection": system.detect_objects(temp_path), "analysis_time": 0}
        elif analysis_type == 'caption':
            results = {"image_path": temp_path, "caption": system.caption_image(temp_path), "analysis_time": 0}
        elif analysis_type == 'ocr':
            results = {"image_path": temp_path, "text_extraction": system.extract_text(temp_path), "analysis_time": 0}
        else:
            results = {"error": "Invalid analysis_type"}
    except Exception as e:
        results = {"error": f"Analysis failed: {str(e)}"}
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
    
    return JSONResponse(content=results)

# ========== ROOT ENDPOINT ==========
@app.get("/")
def root():
    return {"message": "TraeGPT API is running!"}

# ========== MAIN ==========
if __name__ == "__main__":
    uvicorn.run("traegpt_api:app", host="0.0.0.0", port=8000, reload=True) 