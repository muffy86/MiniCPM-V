# Voice Assistant Agent for MiniCPM-V

A powerful and easy-to-use voice AI assistant built on top of the MiniCPM-o 2.6 model. This agent provides speech-to-speech conversation capabilities with memory, configurable voices, and custom voice creation.

## Features

- 🎙️ **Speech-to-Speech Conversations**: Natural voice conversations with audio input and output
- 🧠 **Conversation Memory**: Maintains context across multiple conversation turns
- 🎭 **Multiple Voice Personalities**: Choose from female, male, or default female voices
- ✨ **Custom Voice Creation**: Create unique voices using natural language instructions
- 🔄 **Multi-modal Input**: Support both text and audio inputs
- 🌐 **REST API**: Easy integration with web applications and services
- 📝 **CLI Interface**: Command-line interface for direct interaction

## Quick Start

### Prerequisites

1. Python 3.8 or higher
2. CUDA-compatible GPU (recommended) or CPU
3. Required dependencies (see Installation)

### Installation

```bash
# Install basic dependencies
pip install torch==2.3.1 transformers==4.44.2 librosa==0.9.0 soundfile==0.12.1 fastapi uvicorn gradio

# Install additional dependencies for full functionality
pip install aiofiles onnxruntime
```

### Basic Usage

#### 1. CLI Interface (Recommended for beginners)

```bash
# Start the CLI interface
python voice_assistant_agent.py

# Example commands:
> text Hello, how are you today?
> voice male
> custom Speak like a friendly radio host
> history
> quit
```

#### 2. Python API

```python
from voice_assistant_agent import VoiceAssistantAgent

# Create a voice assistant
assistant = VoiceAssistantAgent(
    voice_type="female",
    language="en"
)

# Text to speech
result = assistant.process_text_input("Hello! How can I help you today?")
print(f"Assistant: {result['text_response']}")
print(f"Audio saved to: {result['audio_path']}")

# Audio to audio (if you have an audio file)
result = assistant.process_audio_input("user_question.wav")
if result['success']:
    print(f"Assistant: {result['text_response']}")
```

#### 3. REST API Server

```bash
# Start the API server
python voice_assistant_api.py --host 0.0.0.0 --port 8000

# The API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

## Detailed Usage

### Voice Assistant Agent Class

The main `VoiceAssistantAgent` class provides the core functionality:

```python
from voice_assistant_agent import VoiceAssistantAgent

# Initialize with custom settings
assistant = VoiceAssistantAgent(
    model_path="openbmb/MiniCPM-o-2_6",  # Model to use
    device="cuda:0",                      # Device (cuda:0, cpu, etc.)
    voice_type="female",                  # Voice type
    language="en"                         # Language
)
```

### Available Voice Types

- **`female`**: Stable female voice (recommended)
- **`male`**: Stable male voice (recommended)  
- **`default_female`**: More human-like but less stable female voice

### Core Methods

#### Process Text Input
```python
result = assistant.process_text_input("What's the weather like today?")
# Returns: {'text_response': str, 'audio_path': str, 'success': bool}
```

#### Process Audio Input
```python
result = assistant.process_audio_input("user_audio.wav")
# Returns: {'text_response': str, 'audio_path': str, 'success': bool}
```

#### Change Voice Type
```python
assistant.set_voice_type("male")
```

#### Create Custom Voice
```python
result = assistant.create_custom_voice("Speak like a wise professor")
# Returns: {'instruction': str, 'audio_path': str, 'success': bool}
```

#### Conversation Management
```python
# Get conversation history
history = assistant.get_conversation_history()

# Clear conversation history
assistant.clear_history()
```

## REST API Endpoints

### Chat Endpoints

#### POST `/chat/text`
Send a text message to the assistant.

```bash
curl -X POST "http://localhost:8000/chat/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello! How are you?", "voice_type": "female"}'
```

#### POST `/chat/audio`
Send an audio file to the assistant.

```bash
curl -X POST "http://localhost:8000/chat/audio" \
     -F "audio_file=@user_question.wav"
```

### Voice Management

#### POST `/voice/change`
Change the assistant's voice type.

```bash
curl -X POST "http://localhost:8000/voice/change" \
     -H "Content-Type: application/json" \
     -d '{"voice_type": "male"}'
```

#### POST `/voice/custom`
Create a custom voice.

```bash
curl -X POST "http://localhost:8000/voice/custom" \
     -H "Content-Type: application/json" \
     -d '{"instruction": "Speak like a confident news anchor"}'
```

### History Management

#### GET `/history`
Get conversation history.

```bash
curl "http://localhost:8000/history"
```

#### DELETE `/history`
Clear conversation history.

```bash
curl -X DELETE "http://localhost:8000/history"
```

### Status

#### GET `/status`
Get assistant status and configuration.

```bash
curl "http://localhost:8000/status"
```

## Examples

### Example 1: Basic Conversation

```python
from voice_assistant_agent import VoiceAssistantAgent

# Create assistant
assistant = VoiceAssistantAgent(voice_type="female")

# Multi-turn conversation
questions = [
    "What's your name?",
    "What can you help me with?",
    "Tell me a joke",
    "What was the joke about?"  # Tests memory
]

for question in questions:
    result = assistant.process_text_input(question)
    if result['success']:
        print(f"User: {question}")
        print(f"Assistant: {result['text_response']}")
        print(f"Audio: {result['audio_path']}")
        print()
```

### Example 2: Voice Personality Demo

```python
# Test different voices
voices = ["female", "male", "default_female"]
message = "Hello! This is a voice demonstration."

for voice in voices:
    assistant.set_voice_type(voice)
    result = assistant.process_text_input(message)
    print(f"{voice.title()} voice: {result['audio_path']}")
```

### Example 3: Custom Voice Creation

```python
# Create custom voices
instructions = [
    "Speak like a wise old wizard",
    "Talk like an excited sports commentator",
    "Sound like a calm meditation teacher"
]

for instruction in instructions:
    result = assistant.create_custom_voice(instruction)
    if result['success']:
        print(f"Created: {instruction}")
        print(f"Sample: {result['audio_path']}")
```

## Advanced Configuration

### Device Selection

```python
# Use specific GPU
assistant = VoiceAssistantAgent(device="cuda:0")

# Use CPU (slower but works without GPU)
assistant = VoiceAssistantAgent(device="cpu")

# Auto-select device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assistant = VoiceAssistantAgent(device=device)
```

### Custom Model Path

```python
# Use local model
assistant = VoiceAssistantAgent(
    model_path="/path/to/local/model",
    device="cuda:0"
)

# Use different model version
assistant = VoiceAssistantAgent(
    model_path="openbmb/MiniCPM-o-2_6-int4",  # Quantized version
    device="cuda:0"
)
```

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Ensure you have sufficient GPU memory (8GB+ recommended)
   - Try using CPU mode: `device="cpu"`
   - Use quantized model: `model_path="openbmb/MiniCPM-o-2_6-int4"`

2. **Audio files not found**
   - Check that audio reference files exist in `./assets/input_examples/`
   - Download missing files from the repository

3. **CUDA out of memory**
   - Reduce model precision or use CPU
   - Clear conversation history frequently: `assistant.clear_history()`

4. **Poor audio quality**
   - Ensure input audio is 16kHz, mono WAV format
   - Try different voice types
   - Check microphone quality

### Performance Tips

- Use CUDA if available for faster inference
- Clear conversation history periodically to save memory
- Use quantized models for better performance on limited hardware
- Batch multiple requests when possible

## File Structure

```
MiniCPM-V/
├── voice_assistant_agent.py      # Main agent class
├── voice_assistant_api.py        # REST API server
├── VOICE_ASSISTANT_README.md     # This documentation
├── examples/
│   ├── basic_voice_assistant.py  # Basic usage examples
│   └── advanced_voice_assistant.py # Advanced features demo
└── assets/input_examples/         # Reference audio files
    ├── assistant_female_voice.wav
    ├── assistant_male_voice.wav
    └── assistant_default_female_voice.wav
```

## Integration Examples

### Web Application Integration

```javascript
// JavaScript example for web integration
async function chatWithAssistant(message) {
    const response = await fetch('/chat/text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: message, voice_type: 'female'})
    });
    
    const result = await response.json();
    if (result.success) {
        console.log('Assistant:', result.data.text_response);
        // Play audio file: result.data.audio_path
    }
}
```

### Discord Bot Integration

```python
import discord
from voice_assistant_agent import VoiceAssistantAgent

class VoiceBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.assistant = VoiceAssistantAgent()
    
    async def on_message(self, message):
        if message.author == self.user:
            return
            
        result = self.assistant.process_text_input(message.content)
        if result['success']:
            await message.channel.send(result['text_response'])
            # Send audio file if needed
```

## Contributing

We welcome contributions! Please see the main repository for contribution guidelines.

## License

This project follows the same license as the main MiniCPM-V repository.

## Support

For issues and questions:
1. Check this documentation
2. Look at the examples in the `examples/` directory
3. Create an issue in the main repository
4. Join the community Discord server

---

*Built with ❤️ using MiniCPM-o 2.6*