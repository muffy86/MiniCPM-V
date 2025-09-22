#!/usr/bin/env python3
"""
Voice Assistant API Server

This module provides a REST API server for the Voice Assistant Agent,
making it easy to integrate voice AI capabilities into web applications
and other services.
"""

import os
import sys
import asyncio
import base64
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Import the voice assistant agent
from voice_assistant_agent import VoiceAssistantAgent

# Pydantic models for API requests
class TextRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "female"
    language: Optional[str] = "en"

class VoiceChangeRequest(BaseModel):
    voice_type: str

class CustomVoiceRequest(BaseModel):
    instruction: str

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Global voice assistant instance
voice_assistant = None

# FastAPI app
app = FastAPI(
    title="Voice Assistant API",
    description="REST API for MiniCPM-o Voice Assistant Agent",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize the voice assistant on startup."""
    global voice_assistant
    
    try:
        print("Initializing Voice Assistant...")
        voice_assistant = VoiceAssistantAgent(
            model_path="openbmb/MiniCPM-o-2_6",
            device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
            voice_type="female",
            language="en"
        )
        print("✓ Voice Assistant initialized successfully!")
        
    except Exception as e:
        print(f"Failed to initialize Voice Assistant: {e}")
        sys.exit(1)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Voice Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat/text": "Send text message to assistant",
            "POST /chat/audio": "Send audio file to assistant",
            "POST /voice/change": "Change voice type",
            "POST /voice/custom": "Create custom voice",
            "GET /history": "Get conversation history",
            "DELETE /history": "Clear conversation history",
            "GET /status": "Get assistant status"
        }
    }

@app.post("/chat/text", response_model=APIResponse)
async def chat_with_text(request: TextRequest):
    """
    Chat with the assistant using text input.
    
    Args:
        request: TextRequest containing the text message
        
    Returns:
        APIResponse with assistant's text response and audio file path
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    try:
        # Change voice type if different from current
        if request.voice_type != voice_assistant.voice_type:
            voice_assistant.set_voice_type(request.voice_type)
        
        # Process the text input
        result = voice_assistant.process_text_input(request.text)
        
        if result['success']:
            return APIResponse(
                success=True,
                message="Response generated successfully",
                data={
                    "text_response": result['text_response'],
                    "audio_path": result['audio_path'],
                    "voice_type": voice_assistant.voice_type
                }
            )
        else:
            return APIResponse(
                success=False,
                message=f"Failed to process text: {result['text_response']}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/chat/audio", response_model=APIResponse)
async def chat_with_audio(audio_file: UploadFile = File(...)):
    """
    Chat with the assistant using audio input.
    
    Args:
        audio_file: Uploaded audio file
        
    Returns:
        APIResponse with assistant's text response and audio file path
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    # Validate file type
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_audio_{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Process the audio input
        result = voice_assistant.process_audio_input(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result['success']:
            return APIResponse(
                success=True,
                message="Audio processed successfully",
                data={
                    "text_response": result['text_response'],
                    "audio_path": result['audio_path'],
                    "voice_type": voice_assistant.voice_type
                }
            )
        else:
            return APIResponse(
                success=False,
                message=f"Failed to process audio: {result['text_response']}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/voice/change", response_model=APIResponse)
async def change_voice(request: VoiceChangeRequest):
    """
    Change the assistant's voice type.
    
    Args:
        request: VoiceChangeRequest containing the new voice type
        
    Returns:
        APIResponse confirming the voice change
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    valid_voices = ["female", "male", "default_female"]
    if request.voice_type not in valid_voices:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid voice type. Must be one of: {valid_voices}"
        )
    
    try:
        voice_assistant.set_voice_type(request.voice_type)
        
        return APIResponse(
            success=True,
            message=f"Voice changed to {request.voice_type}",
            data={"current_voice": voice_assistant.voice_type}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/voice/custom", response_model=APIResponse)
async def create_custom_voice(request: CustomVoiceRequest):
    """
    Create a custom voice based on instruction.
    
    Args:
        request: CustomVoiceRequest containing voice instruction
        
    Returns:
        APIResponse with custom voice audio file path
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    try:
        result = voice_assistant.create_custom_voice(request.instruction)
        
        if result['success']:
            return APIResponse(
                success=True,
                message="Custom voice created successfully",
                data={
                    "instruction": result['instruction'],
                    "audio_path": result['audio_path']
                }
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to create custom voice"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/history", response_model=APIResponse)
async def get_conversation_history():
    """
    Get the current conversation history.
    
    Returns:
        APIResponse with conversation history
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    try:
        history = voice_assistant.get_conversation_history()
        
        # Convert history to JSON-serializable format
        serializable_history = []
        for msg in history:
            serializable_msg = {
                "role": msg["role"],
                "content": str(msg["content"])[:200] + "..." if len(str(msg["content"])) > 200 else str(msg["content"])
            }
            serializable_history.append(serializable_msg)
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(history)} conversation messages",
            data={
                "history": serializable_history,
                "total_messages": len(history)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.delete("/history", response_model=APIResponse)
async def clear_conversation_history():
    """
    Clear the conversation history.
    
    Returns:
        APIResponse confirming history cleared
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    try:
        voice_assistant.clear_history()
        
        return APIResponse(
            success=True,
            message="Conversation history cleared successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/status", response_model=APIResponse)
async def get_status():
    """
    Get the current status of the voice assistant.
    
    Returns:
        APIResponse with assistant status information
    """
    global voice_assistant
    
    if not voice_assistant:
        raise HTTPException(status_code=503, detail="Voice assistant not initialized")
    
    try:
        history_count = len(voice_assistant.get_conversation_history())
        
        return APIResponse(
            success=True,
            message="Voice assistant is running",
            data={
                "model_path": voice_assistant.model_path,
                "device": voice_assistant.device,
                "voice_type": voice_assistant.voice_type,
                "language": voice_assistant.language,
                "conversation_messages": history_count
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/audio/{audio_filename}")
async def get_audio_file(audio_filename: str):
    """
    Download an audio response file.
    
    Args:
        audio_filename: Name of the audio file to download
        
    Returns:
        FileResponse with the audio file
    """
    file_path = audio_filename
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type='audio/wav',
        filename=audio_filename
    )

def main():
    """Main function to start the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Assistant API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("Starting Voice Assistant API Server...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print("API documentation will be available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "voice_assistant_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()