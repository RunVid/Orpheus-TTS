import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import io
import time
import numpy as np
import multiprocessing
import threading
import uuid
import os
import gc

# Global variables
global_model = None
model_lock = threading.Lock()
output_dir = "outputs"

def load_model(model_path):
    """Load the Orpheus TTS model once and return it"""
    print(f"Loading Orpheus TTS model from: {model_path}")
    start_time = time.monotonic()
    model = OrpheusModel(model_name=model_path)
    end_time = time.monotonic()
    print(f"Model loaded in {end_time - start_time:.2f} seconds")
    return model

def generate_speech(prompt, voice="tara"):
    """Generate speech using the globally loaded model with lock protection"""
    global global_model
    
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())
    print(f"Processing request {request_id} for voice: {voice}")
    
    start_time = time.monotonic()
    
    # Acquire lock to ensure only one request at a time
    with model_lock:
        try:
            # Generate speech
            print(f"Starting generation for request {request_id}")
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
            
            # Reset buffer position
            wav_buffer.seek(0)
            
            # Force garbage collection after generation
            gc.collect()
            
            end_time = time.monotonic()
            generation_time = end_time - start_time
            print(f"Generation completed for request {request_id} in {generation_time:.2f} seconds")
            
            return wav_buffer, duration, generation_time
            
        except Exception as e:
            print(f"Error in request {request_id}: {str(e)}")
            raise e

def create_interface(model_path):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
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
                status_msg = gr.Textbox(label="Status", value="Ready", interactive=False)
                
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
                # Create a unique filename for each request to avoid conflicts
                unique_id = str(uuid.uuid4())[:8]
                temp_filename = os.path.join(output_dir, f"output_{unique_id}.wav")
                
                # Update status
                yield temp_filename, "Processing...", "Processing..."
                
                # Generate the speech
                wav_buffer, duration, gen_time = generate_speech(text_input, voice)
                
                # Save to file for Gradio to display
                with open(temp_filename, "wb") as f:
                    f.write(wav_buffer.getvalue())
                
                # Return results
                yield temp_filename, f"{duration:.2f}", f"{gen_time:.2f}"
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                yield None, "Error", error_msg
        
        generate_btn.click(
            fn=process_text,
            inputs=[text_input, voice],
            outputs=[audio_output, duration_output, generation_time],
            show_progress=True
        ).then(
            fn=lambda: "Ready",
            inputs=None,
            outputs=status_msg
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
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run the Gradio server on (default: 7860)')
    parser.add_argument('--share', action='store_true', 
                        help='Create a shareable link for the interface')
    args = parser.parse_args()
    
    # Load the model once at startup
    print("Initializing Orpheus TTS model...")
    global_model = load_model(args.model_path)
    
    # Create and launch the interface
    print("Starting Gradio web interface...")
    app = create_interface(args.model_path)
    app.queue(concurrency_count=1).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    ) 