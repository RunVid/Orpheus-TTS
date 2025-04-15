import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import io
import time
import numpy as np
import multiprocessing

# Global variable to store the loaded model
global_model = None

def load_model(model_path):
    """Load the Orpheus TTS model once and return it"""
    print(f"Loading Orpheus TTS model from: {model_path}")
    start_time = time.monotonic()
    model = OrpheusModel(model_name=model_path)
    end_time = time.monotonic()
    print(f"Model loaded in {end_time - start_time:.2f} seconds")
    return model

def generate_speech(prompt, voice="tara"):
    """Generate speech using the globally loaded model"""
    global global_model
    
    start_time = time.monotonic()
    
    # Generate speech
    syn_tokens = global_model.generate_speech(
        prompt=prompt,
        voice=voice,
    )
    
    # Create in-memory wave file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        
        duration = total_frames / wf.getframerate()
    
    end_time = time.monotonic()
    generation_time = end_time - start_time
    
    # Reset buffer position
    wav_buffer.seek(0)
    
    # Return audio data and generation stats
    return wav_buffer, duration, generation_time

def create_interface(model_path):
    # Default values
    default_prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
    
    with gr.Blocks(title="Orpheus TTS Demo") as demo:
        gr.Markdown("# Orpheus TTS Web Interface")
        gr.Markdown(f"### Using model: {model_path}")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    value=default_prompt,
                    lines=5,
                    placeholder="Enter text to convert to speech..."
                )
                
                voice = gr.Dropdown(
                    label="Voice",
                    choices=["tara", "emma", "bella", "antoni", "josh", "michael"],
                    value="tara"
                )
                
                generate_btn = gr.Button("Generate Speech", variant="primary")
                
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="filepath",
                    interactive=False
                )
                
                with gr.Row():
                    duration_output = gr.Textbox(label="Audio Duration (seconds)")
                    generation_time = gr.Textbox(label="Generation Time (seconds)")
        
        def process_text(text_input, voice):
            try:
                wav_buffer, duration, gen_time = generate_speech(text_input, voice)
                
                # Save to temporary file for Gradio to display
                temp_filename = "output.wav"
                with open(temp_filename, "wb") as f:
                    f.write(wav_buffer.getvalue())
                
                return temp_filename, f"{duration:.2f}", f"{gen_time:.2f}"
            except Exception as e:
                return None, "Error", f"Error: {str(e)}"
        
        generate_btn.click(
            fn=process_text,
            inputs=[text_input, voice],
            outputs=[audio_output, duration_output, generation_time]
        )
    
    return demo

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Default model path
    default_model_path = "/home/ubuntu/models/orpheus-3b-0.1-ft"
    
    # Command line arguments for model path
    import argparse
    parser = argparse.ArgumentParser(description='Orpheus TTS Gradio Interface')
    parser.add_argument('--model_path', type=str, default=default_model_path,
                        help=f'Path to the Orpheus TTS model (default: {default_model_path})')
    args = parser.parse_args()
    
    # Load the model once at startup
    print("Initializing Orpheus TTS model...")
    global_model = load_model(args.model_path)
    
    # Create and launch the interface
    print("Starting Gradio web interface...")
    app = create_interface(args.model_path)
    app.launch(share=True) 