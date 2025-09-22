"""
Voice AI Assistant Agent for MiniCPM-o 2.6

This module provides a simple interface to create and interact with a voice AI assistant
using the MiniCPM-o 2.6 model capabilities.
"""

import os
import torch
import librosa
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer

class VoiceAssistantAgent:
    """
    A voice AI assistant agent that leverages MiniCPM-o 2.6 for speech conversation.
    
    This class provides an easy-to-use interface for creating voice assistants with
    different personalities and managing conversation history.
    """
    
    def __init__(self, 
                 model_path: str = "openbmb/MiniCPM-o-2_6",
                 device: str = "cuda:0",
                 voice_type: str = "female",
                 language: str = "en"):
        """
        Initialize the Voice Assistant Agent.
        
        Args:
            model_path: Path to the MiniCPM-o model
            device: Device to run the model on (cuda:0, cpu, etc.)
            voice_type: Type of voice - "female", "male", or "default_female"
            language: Language for the assistant (en, zh, etc.)
        """
        self.model_path = model_path
        self.device = device
        self.voice_type = voice_type
        self.language = language
        self.conversation_history = []
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Load model and tokenizer
        self._load_model()
        
        # Set up voice reference
        self._setup_voice_reference()
        
        # Initialize system prompt
        self._initialize_system_prompt()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the assistant."""
        logger = logging.getLogger("voice_assistant")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Load the MiniCPM-o model and tokenizer."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation='sdpa'
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Initialize TTS
            self.model.init_tts()
            self.model.to(self.device).eval()
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_voice_reference(self):
        """Set up the reference voice based on voice_type."""
        voice_paths = {
            "female": "./assets/input_examples/assistant_female_voice.wav",
            "male": "./assets/input_examples/assistant_male_voice.wav", 
            "default_female": "./assets/input_examples/assistant_default_female_voice.wav"
        }
        
        if self.voice_type not in voice_paths:
            self.logger.warning(f"Unknown voice type {self.voice_type}, using female")
            self.voice_type = "female"
        
        self.ref_audio_path = voice_paths[self.voice_type]
        
        # Check if reference audio exists
        if not os.path.exists(self.ref_audio_path):
            self.logger.warning(f"Reference audio not found at {self.ref_audio_path}")
            # Use a fallback or create a simple reference
        
        self.logger.info(f"Using {self.voice_type} voice reference")
    
    def _initialize_system_prompt(self):
        """Initialize the system prompt for the assistant."""
        try:
            # Load reference audio
            ref_audio, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            
            # Get system prompt using the model's built-in method
            self.sys_prompt = self.model.get_sys_prompt(
                ref_audio=ref_audio,
                mode='audio_assistant',
                language=self.language
            )
            
            self.logger.info("System prompt initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system prompt: {e}")
            # Create a fallback system prompt
            self.sys_prompt = {
                'role': 'system',
                'content': 'You are a helpful AI assistant that can understand and respond with speech.'
            }
    
    def process_audio_input(self, audio_path: str) -> Dict[str, Any]:
        """
        Process audio input and generate response.
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            Dictionary containing the response text and audio path
        """
        try:
            # Load user audio
            user_audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            
            # Create user message
            user_message = {
                'role': 'user',
                'content': [user_audio]
            }
            
            # Prepare messages (system prompt + history + current message)
            msgs = [self.sys_prompt] + self.conversation_history + [user_message]
            
            # Generate response
            output_audio_path = f"response_{len(self.conversation_history)}.wav"
            
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                max_new_tokens=128,
                use_tts_template=True,
                generate_audio=True,
                temperature=0.3,
                output_audio_path=output_audio_path,
            )
            
            # Add to conversation history
            self.conversation_history.extend([
                user_message,
                {'role': 'assistant', 'content': response}
            ])
            
            self.logger.info(f"Generated response: {response}")
            
            return {
                'text_response': response,
                'audio_path': output_audio_path,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process audio input: {e}")
            return {
                'text_response': f"Sorry, I encountered an error: {str(e)}",
                'audio_path': None,
                'success': False
            }
    
    def process_text_input(self, text: str) -> Dict[str, Any]:
        """
        Process text input and generate speech response.
        
        Args:
            text: Input text from user
            
        Returns:
            Dictionary containing the response text and audio path
        """
        try:
            # Create user message with text
            user_message = {
                'role': 'user', 
                'content': [text]
            }
            
            # Prepare messages
            msgs = [self.sys_prompt] + self.conversation_history + [user_message]
            
            # Generate response
            output_audio_path = f"response_{len(self.conversation_history)}.wav"
            
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                max_new_tokens=128,
                use_tts_template=True,
                generate_audio=True,
                temperature=0.3,
                output_audio_path=output_audio_path,
            )
            
            # Add to conversation history
            self.conversation_history.extend([
                user_message,
                {'role': 'assistant', 'content': response}
            ])
            
            self.logger.info(f"Generated response: {response}")
            
            return {
                'text_response': response,
                'audio_path': output_audio_path,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process text input: {e}")
            return {
                'text_response': f"Sorry, I encountered an error: {str(e)}",
                'audio_path': None,
                'success': False
            }
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def set_voice_type(self, voice_type: str):
        """
        Change the voice type of the assistant.
        
        Args:
            voice_type: New voice type - "female", "male", or "default_female"
        """
        if voice_type != self.voice_type:
            self.voice_type = voice_type
            self._setup_voice_reference()
            self._initialize_system_prompt()
            self.logger.info(f"Voice type changed to {voice_type}")
    
    def create_custom_voice(self, instruction: str) -> Dict[str, Any]:
        """
        Create a custom voice based on instruction (Instruction-to-Speech).
        
        Args:
            instruction: Description of the desired voice
            
        Returns:
            Dictionary containing the generated audio path
        """
        try:
            msgs = [{'role': 'user', 'content': [instruction]}]
            
            output_audio_path = "custom_voice_sample.wav"
            
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                max_new_tokens=128,
                use_tts_template=True,
                generate_audio=True,
                temperature=0.3,
                output_audio_path=output_audio_path,
            )
            
            self.logger.info(f"Custom voice created: {instruction}")
            
            return {
                'instruction': instruction,
                'response': response,
                'audio_path': output_audio_path,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create custom voice: {e}")
            return {
                'instruction': instruction,
                'response': None,
                'audio_path': None,
                'success': False
            }


class VoiceAssistantCLI:
    """
    Command Line Interface for the Voice Assistant Agent.
    """
    
    def __init__(self, agent: VoiceAssistantAgent):
        self.agent = agent
        self.running = True
    
    def print_help(self):
        """Print help information."""
        help_text = """
Voice Assistant CLI Commands:
- 'text <message>': Send a text message to the assistant
- 'audio <path>': Send an audio file to the assistant  
- 'voice <type>': Change voice type (female/male/default_female)
- 'custom <instruction>': Create custom voice with instruction
- 'history': Show conversation history
- 'clear': Clear conversation history
- 'help': Show this help message
- 'quit': Exit the assistant

Example usage:
> text Hello, how are you today?
> audio my_question.wav
> voice male
> custom Speak like a confident news anchor
        """
        print(help_text)
    
    def run(self):
        """Run the CLI interface."""
        print("Voice Assistant Agent Started!")
        print("Type 'help' for commands or 'quit' to exit.")
        print(f"Current voice: {self.agent.voice_type}")
        print("-" * 50)
        
        while self.running:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                
                if command == 'quit':
                    self.running = False
                    print("Goodbye!")
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'text' and len(parts) > 1:
                    message = parts[1]
                    print(f"Processing text: {message}")
                    result = self.agent.process_text_input(message)
                    
                    if result['success']:
                        print(f"Assistant: {result['text_response']}")
                        print(f"Audio saved to: {result['audio_path']}")
                    else:
                        print(f"Error: {result['text_response']}")
                
                elif command == 'audio' and len(parts) > 1:
                    audio_path = parts[1]
                    if os.path.exists(audio_path):
                        print(f"Processing audio: {audio_path}")
                        result = self.agent.process_audio_input(audio_path)
                        
                        if result['success']:
                            print(f"Assistant: {result['text_response']}")
                            print(f"Audio response saved to: {result['audio_path']}")
                        else:
                            print(f"Error: {result['text_response']}")
                    else:
                        print(f"Audio file not found: {audio_path}")
                
                elif command == 'voice' and len(parts) > 1:
                    voice_type = parts[1]
                    if voice_type in ['female', 'male', 'default_female']:
                        self.agent.set_voice_type(voice_type)
                        print(f"Voice changed to: {voice_type}")
                    else:
                        print("Invalid voice type. Use: female, male, or default_female")
                
                elif command == 'custom' and len(parts) > 1:
                    instruction = parts[1]
                    print(f"Creating custom voice: {instruction}")
                    result = self.agent.create_custom_voice(instruction)
                    
                    if result['success']:
                        print(f"Custom voice created!")
                        print(f"Audio sample saved to: {result['audio_path']}")
                    else:
                        print("Failed to create custom voice")
                
                elif command == 'history':
                    history = self.agent.get_conversation_history()
                    if history:
                        print("\nConversation History:")
                        for i, msg in enumerate(history):
                            role = msg['role'].title()
                            content = str(msg['content'])[:100] + "..." if len(str(msg['content'])) > 100 else str(msg['content'])
                            print(f"{i+1}. {role}: {content}")
                    else:
                        print("No conversation history")
                
                elif command == 'clear':
                    self.agent.clear_history()
                    print("Conversation history cleared")
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                self.running = False
                print("\nGoodbye!")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Assistant Agent CLI")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-2_6", help="Model path")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--voice", default="female", choices=["female", "male", "default_female"], help="Voice type")
    parser.add_argument("--language", default="en", help="Language")
    
    args = parser.parse_args()
    
    try:
        # Create voice assistant agent
        agent = VoiceAssistantAgent(
            model_path=args.model,
            device=args.device,
            voice_type=args.voice,
            language=args.language
        )
        
        # Start CLI
        cli = VoiceAssistantCLI(agent)
        cli.run()
        
    except Exception as e:
        print(f"Failed to start voice assistant: {e}")
        print("Make sure you have the required dependencies installed and the model is available.")