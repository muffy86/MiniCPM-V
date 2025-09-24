#!/usr/bin/env python3
"""
Advanced Voice Assistant Example

This example demonstrates advanced features of the voice assistant including
audio file processing and context-aware conversations.
"""

import sys
import os
import time

# Add the parent directory to the path to import voice_assistant_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant_agent import VoiceAssistantAgent

def simulate_audio_conversation(assistant, audio_files):
    """
    Simulate an audio conversation using provided audio files.
    
    Args:
        assistant: VoiceAssistantAgent instance
        audio_files: List of audio file paths
    """
    print("Audio Conversation Simulation")
    print("-" * 40)
    
    for i, audio_file in enumerate(audio_files, 1):
        if os.path.exists(audio_file):
            print(f"Turn {i}: Processing audio file: {audio_file}")
            result = assistant.process_audio_input(audio_file)
            
            if result['success']:
                print(f"Turn {i}: Assistant responded: {result['text_response']}")
                print(f"Turn {i}: Audio response saved to: {result['audio_path']}")
            else:
                print(f"Turn {i}: Error: {result['text_response']}")
        else:
            print(f"Turn {i}: Audio file not found: {audio_file}")
            # Fallback to text input
            text_input = f"This is simulated text input for turn {i}"
            print(f"Turn {i}: Using text fallback: {text_input}")
            result = assistant.process_text_input(text_input)
            
            if result['success']:
                print(f"Turn {i}: Assistant responded: {result['text_response']}")
        
        print()
        time.sleep(1)  # Small delay between turns

def demonstrate_context_awareness(assistant):
    """
    Demonstrate how the assistant maintains context across conversation turns.
    
    Args:
        assistant: VoiceAssistantAgent instance
    """
    print("Context Awareness Demonstration")
    print("-" * 40)
    
    # Context-building conversation
    context_conversation = [
        "My name is Alice and I'm a software engineer.",
        "I'm working on a machine learning project about natural language processing.",
        "What do you think about my project?",
        "Can you suggest some improvements?",
        "What's my name again?",  # Test if assistant remembers
        "What was I working on?",  # Test project memory
    ]
    
    for i, message in enumerate(context_conversation, 1):
        print(f"Turn {i} - User: {message}")
        result = assistant.process_text_input(message)
        
        if result['success']:
            print(f"Turn {i} - Assistant: {result['text_response']}")
        else:
            print(f"Turn {i} - Error: {result['text_response']}")
        print()
        time.sleep(0.5)

def demonstrate_voice_personalities(assistant):
    """
    Demonstrate different voice personalities and custom voices.
    
    Args:
        assistant: VoiceAssistantAgent instance
    """
    print("Voice Personalities Demonstration")
    print("-" * 40)
    
    # Test different voice types
    voice_types = ["female", "male", "default_female"]
    test_message = "Hello! I'm demonstrating different voice personalities."
    
    for voice_type in voice_types:
        print(f"Testing {voice_type} voice:")
        assistant.set_voice_type(voice_type)
        
        result = assistant.process_text_input(test_message)
        if result['success']:
            print(f"  Response: {result['text_response']}")
            print(f"  Audio saved to: {result['audio_path']}")
        print()
    
    # Test custom voices
    print("Testing custom voice personalities:")
    custom_voices = [
        "Speak like a wise old professor giving a lecture",
        "Talk like an excited sports commentator",
        "Sound like a calm meditation guide",
        "Speak like a friendly children's storyteller"
    ]
    
    for i, instruction in enumerate(custom_voices, 1):
        print(f"Custom voice {i}: {instruction}")
        result = assistant.create_custom_voice(instruction)
        
        if result['success']:
            print(f"  Custom voice created! Audio: {result['audio_path']}")
        else:
            print(f"  Failed to create custom voice")
        print()

def performance_test(assistant, num_iterations=5):
    """
    Run a simple performance test on the voice assistant.
    
    Args:
        assistant: VoiceAssistantAgent instance
        num_iterations: Number of test iterations
    """
    print(f"Performance Test ({num_iterations} iterations)")
    print("-" * 40)
    
    test_messages = [
        "What time is it?",
        "Tell me a fun fact",
        "How's the weather?",
        "What can you help me with?",
        "Goodbye!"
    ]
    
    total_time = 0
    successful_responses = 0
    
    for i in range(num_iterations):
        message = test_messages[i % len(test_messages)]
        
        start_time = time.time()
        result = assistant.process_text_input(f"{message} (Test {i+1})")
        end_time = time.time()
        
        response_time = end_time - start_time
        total_time += response_time
        
        if result['success']:
            successful_responses += 1
            print(f"Test {i+1}: ✓ Response time: {response_time:.2f}s")
        else:
            print(f"Test {i+1}: ✗ Failed after {response_time:.2f}s")
    
    print()
    print("Performance Summary:")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Successful responses: {successful_responses}")
    print(f"  Success rate: {(successful_responses/num_iterations)*100:.1f}%")
    print(f"  Average response time: {total_time/num_iterations:.2f}s")
    print()

def main():
    """Main function to run advanced voice assistant examples."""
    
    print("=== Advanced Voice Assistant Example ===")
    print("This example demonstrates advanced features and capabilities.")
    print()
    
    try:
        # Create a voice assistant
        print("Initializing advanced voice assistant...")
        assistant = VoiceAssistantAgent(
            model_path="openbmb/MiniCPM-o-2_6",
            device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
            voice_type="female",
            language="en"
        )
        print("✓ Voice assistant initialized successfully!")
        print()
        
        # Example 1: Context awareness
        demonstrate_context_awareness(assistant)
        
        # Clear history for next example
        assistant.clear_history()
        print("Conversation history cleared for next example.\n")
        
        # Example 2: Voice personalities
        demonstrate_voice_personalities(assistant)
        
        # Clear history for next example
        assistant.clear_history()
        print("Conversation history cleared for next example.\n")
        
        # Example 3: Audio file conversation (with fallback)
        audio_files = [
            "./assets/input_examples/audio_understanding.mp3",  # May not exist
            "./test_audio_1.wav",  # May not exist
            "./test_audio_2.wav",  # May not exist
        ]
        simulate_audio_conversation(assistant, audio_files)
        
        # Clear history for performance test
        assistant.clear_history()
        
        # Example 4: Performance test
        performance_test(assistant, num_iterations=3)  # Reduced for demo
        
        print("=== Advanced examples completed! ===")
        print()
        print("Summary of features demonstrated:")
        print("✓ Context-aware conversations")
        print("✓ Multiple voice personalities") 
        print("✓ Custom voice creation")
        print("✓ Audio file processing (with fallback)")
        print("✓ Performance testing")
        print("✓ Conversation history management")
        
    except Exception as e:
        print(f"Error running advanced examples: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure all dependencies are installed")
        print("2. Check CUDA availability")
        print("3. Verify model access")
        print("4. Make sure you have sufficient memory")

if __name__ == "__main__":
    main()