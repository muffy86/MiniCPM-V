#!/usr/bin/env python3
"""
Basic Voice Assistant Example

This example demonstrates how to create and use a basic voice assistant
using the VoiceAssistantAgent class.
"""

import sys
import os

# Add the parent directory to the path to import voice_assistant_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant_agent import VoiceAssistantAgent

def main():
    """Main function to demonstrate basic voice assistant usage."""
    
    print("=== Basic Voice Assistant Example ===")
    print("This example shows how to create and interact with a voice assistant.")
    print()
    
    try:
        # Create a voice assistant with female voice
        print("Creating voice assistant with female voice...")
        assistant = VoiceAssistantAgent(
            model_path="openbmb/MiniCPM-o-2_6",
            device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
            voice_type="female",
            language="en"
        )
        print("✓ Voice assistant created successfully!")
        print()
        
        # Example 1: Text input with speech output
        print("Example 1: Text input -> Speech output")
        print("-" * 40)
        
        text_message = "Hello! How are you doing today?"
        print(f"User: {text_message}")
        
        result = assistant.process_text_input(text_message)
        
        if result['success']:
            print(f"Assistant: {result['text_response']}")
            print(f"Audio response saved to: {result['audio_path']}")
        else:
            print(f"Error: {result['text_response']}")
        
        print()
        
        # Example 2: Multiple conversation turns
        print("Example 2: Multi-turn conversation")
        print("-" * 40)
        
        conversations = [
            "What's the weather like?",
            "Can you tell me a joke?", 
            "What's your favorite color?"
        ]
        
        for i, message in enumerate(conversations, 1):
            print(f"Turn {i} - User: {message}")
            result = assistant.process_text_input(message)
            
            if result['success']:
                print(f"Turn {i} - Assistant: {result['text_response']}")
                print(f"Audio saved to: {result['audio_path']}")
            else:
                print(f"Turn {i} - Error: {result['text_response']}")
            print()
        
        # Example 3: Change voice type
        print("Example 3: Changing voice type")
        print("-" * 40)
        
        print("Changing to male voice...")
        assistant.set_voice_type("male")
        
        result = assistant.process_text_input("Hello, this is my new voice!")
        
        if result['success']:
            print(f"Assistant (male voice): {result['text_response']}")
            print(f"Audio saved to: {result['audio_path']}")
        
        print()
        
        # Example 4: Custom voice creation
        print("Example 4: Custom voice creation")
        print("-" * 40)
        
        custom_instruction = "Speak like a friendly radio host with enthusiasm and energy"
        print(f"Creating custom voice with instruction: {custom_instruction}")
        
        result = assistant.create_custom_voice(custom_instruction)
        
        if result['success']:
            print(f"Custom voice created successfully!")
            print(f"Sample audio saved to: {result['audio_path']}")
        else:
            print("Failed to create custom voice")
        
        print()
        
        # Example 5: Conversation history
        print("Example 5: Viewing conversation history")
        print("-" * 40)
        
        history = assistant.get_conversation_history()
        print(f"Total conversation turns: {len(history)}")
        
        for i, msg in enumerate(history[-4:], 1):  # Show last 4 messages
            role = msg['role'].title()
            content = str(msg['content'])
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"{i}. {role}: {content}")
        
        print()
        print("=== Example completed successfully! ===")
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have:")
        print("1. Installed all required dependencies")
        print("2. CUDA available (or use CPU)")
        print("3. Access to the MiniCPM-o-2_6 model")

if __name__ == "__main__":
    main()