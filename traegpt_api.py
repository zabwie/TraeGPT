import os
import sys
import io
import json
import time
import datetime
import tempfile
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import traceback

# Import the core logic from traegpt.py
from traegpt import (
    ImageRecognitionSystem,
    SYSTEM_PROMPT,
    MODEL,
    OLLAMA_URL,
    duckduckgo_search_web,
    google_search,
    get_page_text,
    openrouter_chat
)
import requests

# ========== UTILITY FUNCTIONS ==========
def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # Handle other numpy objects
        return obj.tolist()
    return obj

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

# ========== GLOBAL MEMORY ==========
memory = {}

# ========== CHAT ENDPOINT (OpenAI-compatible) ==========
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(request: ChatRequest):
    try:
        # Check if user is asking about the last image
        last_user_message = request.messages[-1].content.lower()
        if any(phrase in last_user_message for phrase in ["last photo", "last image", "previous photo", "previous image"]):
            if "last_image" in memory:
                label = None
                classification = memory["last_image"].get("classification")
                if isinstance(classification, list) and len(classification) > 0:
                    label = classification[0].get("class")
                elif isinstance(classification, dict):
                    label = classification.get("class")
                else:
                    objects = memory["last_image"].get("objects")
                    if isinstance(objects, list) and len(objects) > 0:
                        label = objects[0].get("class")
                    elif isinstance(objects, dict):
                        label = objects.get("class")
                caption = memory["last_image"].get("caption")
                if label or caption:
                    response = f"It looks like: {label}." if label else ""
                    if caption:
                        response += f" Caption: {caption}"
                else:
                    response = "Sorry, I couldn't confidently recognize the image."
                return ChatCompletionResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=MODEL,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response),
                        finish_reason="stop"
                    )]
                )
            else:
                response = "I don't remember the last image."
                return ChatCompletionResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=MODEL,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response),
                        finish_reason="stop"
                    )]
                )
        # Compose prompt from messages
        prompt = SYSTEM_PROMPT
        for msg in request.messages:
            if msg.role == "user":
                prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"AI: {msg.content}\n"
        prompt += "AI:"
        # Call OpenRouter API
        kimi_result = openrouter_chat(prompt)
        if not isinstance(kimi_result, dict):
            raise ValueError(f"openrouter_chat did not return a dict: {repr(kimi_result)}")
        if "choices" not in kimi_result or not kimi_result["choices"]:
            raise ValueError(f"No choices in OpenRouter response: {json.dumps(kimi_result)}")
        ai_response = kimi_result["choices"][0]["message"]["content"]
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
    except Exception as e:
        print("[Kimi API error]", e)
        traceback.print_exc()
        from fastapi.responses import JSONResponse
        return JSONResponse(content={"error": str(e), "traceback": traceback.format_exc()}, status_code=500)

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
        # Save results to memory for context
        memory["last_image"] = {
            "objects": results.get("object_detection"),
            "caption": results.get("caption"),
            "classification": results.get("classification"),
            "text": results.get("text_extraction")
        }
    except Exception as e:
        results = {"error": f"Analysis failed: {str(e)}"}
    finally:
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
    
    return JSONResponse(content=convert_numpy_types(results))

# ========== SEARCH ENDPOINT ==========
@app.post("/v1/search")
def web_search_api(query: str = Body(..., embed=True), num_results: int = Body(3, embed=True)):
    # Try DuckDuckGo first
    ddg_results = duckduckgo_search_web(query, num_results=num_results)
    results = []
    if ddg_results:
        for r in ddg_results:
            preview = get_page_text(r.get('href', ''), max_chars=200)
            results.append({
                "title": r.get('title', ''),
                "url": r.get('href', ''),
                "snippet": r.get('body', ''),
                "preview": preview
            })
    else:
        # Fallback to Google
        google_results = google_search(query, num_results=num_results)
        results.append({"google_results": google_results})
    return JSONResponse(content=results)

# ========== ROOT ENDPOINT ==========
@app.get("/")
def root():
    return {"message": "TraeGPT API is running!"}

# ========== MAIN ==========
if __name__ == "__main__":
    uvicorn.run("traegpt_api:app", host="0.0.0.0", port=8000, reload=True) 