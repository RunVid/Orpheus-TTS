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
import sys

# Global variables
global_model = None
model_lock = threading.Lock()
output_dir = "outputs"
model_load_stats = {}

def load_model(model_path, disable_compile=False):
    """Load the Orpheus TTS model once and return it with detailed timing stats"""
    global model_load_stats
    
    # Disable compilation if requested
    if disable_compile:
        print("Disabling PyTorch compilation for faster loading (may affect inference speed)")
        # These environment variables are based on vLLM documentation
        # See: https://docs.vllm.ai/en/latest/getting_started/configuration.html
        os.environ["VLLM_USE_CUDA_GRAPH"] = "0"
        os.environ["VLLM_DISABLE_TORCH_COMPILE"] = "1"
    
    print(f"Loading Orpheus TTS model from: {model_path}")
    total_start_time = time.monotonic()
    
    # Track initialization time
    init_start = time.monotonic()
    model = OrpheusModel(model_name=model_path)
    init_end = time.monotonic()
    init_time = init_end - init_start
    
    # Warm up the model with a short test generation to ensure everything is loaded
    warmup_start = time.monotonic()
    try:
        print("Performing warm-up generation...")
        test_prompt = "This is a test."
        list(model.generate_speech(prompt=test_prompt, voice="tara"))
        print("Warm-up completed successfully")
    except Exception as e:
        print(f"Warm-up generation failed: {e}")
    warmup_end = time.monotonic()
    warmup_time = warmup_end - warmup_start
    
    # Calculate total loading time
    total_end_time = time.monotonic()
    total_load_time = total_end_time - total_start_time
    
    # Store stats in global dictionary
    model_load_stats = {
        "total_time": total_load_time,
        "init_time": init_time,
        "warmup_time": warmup_time,
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_path": model_path,
        "compilation_disabled": disable_compile
    }
    
    # Print summary
    print(f"Model loading statistics:")
    print(f"  - Initialization time: {init_time:.2f} seconds")
    print(f"  - Warm-up time: {warmup_time:.2f} seconds")
    print(f"  - Total loading time: {total_load_time:.2f} seconds")
    print(f"  - Compilation disabled: {disable_compile}")
    
    return model

def generate_speech(prompt, voice="tara", temperature=0.6, top_p=0.8, max_tokens=1200, 
                    stop_token_ids=[49158], repetition_penalty=1.3):
    """Generate speech using the globally loaded model with lock protection"""
    global global_model
    
    # Create a unique ID for this request
    request_id = str(uuid.uuid4())
    print(f"Processing request {request_id} for voice: {voice}")
    print(f"Generation parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, rep_penalty={repetition_penalty}")
    print(f"Input text length: {len(prompt)} characters")
    
    start_time = time.monotonic()
    
    # Acquire lock to ensure only one request at a time
    with model_lock:
        try:
            # Generate speech
            print(f"Starting generation for request {request_id}")
            
            try:
                syn_tokens = global_model.generate_speech(
                    prompt=prompt,
                    voice=voice,
                    request_id=request_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                    repetition_penalty=repetition_penalty
                )
                
                # Convert generator to list to ensure all tokens are generated
                syn_tokens_list = list(syn_tokens)
                print(f"Generation produced {len(syn_tokens_list)} chunks")
                
                if not syn_tokens_list:
                    raise Exception("No audio chunks were generated")
                
                # Create in-memory wave file
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    
                    total_frames = 0
                    for audio_chunk in syn_tokens_list:
                        if not isinstance(audio_chunk, bytes):
                            print(f"Warning: Expected bytes but got {type(audio_chunk)}")
                            continue
                            
                        frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                        total_frames += frame_count
                        wf.writeframes(audio_chunk)
                    
                    duration = total_frames / wf.getframerate()
                    print(f"Audio duration: {duration:.2f} seconds from {total_frames} frames")
                
                # Reset buffer position
                wav_buffer.seek(0)
                
                # Force garbage collection after generation
                gc.collect()
                
                end_time = time.monotonic()
                generation_time = end_time - start_time
                print(f"Generation completed for request {request_id} in {generation_time:.2f} seconds")
                
                return wav_buffer, duration, generation_time
                
            except Exception as inner_e:
                print(f"Error during generation: {str(inner_e)}")
                print(f"Exception type: {type(inner_e)}")
                import traceback
                traceback.print_exc()
                raise inner_e
            
        except Exception as e:
            print(f"Error in request {request_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

def create_interface(model_path):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default values
    default_prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
    
    with gr.Blocks(title="Orpheus TTS Demo") as demo:
        gr.Markdown("# Orpheus TTS Web Interface")
        gr.Markdown(f"### Using model: {model_path}")
        
        # Model loading statistics
        if model_load_stats:
            with gr.Accordion("Model Loading Statistics", open=False):
                gr.Markdown(f"""
                - **Model Path**: {model_load_stats['model_path']}
                - **Loaded At**: {model_load_stats['loaded_at']}
                - **Initialization Time**: {model_load_stats['init_time']:.2f} seconds
                - **Warm-up Time**: {model_load_stats['warmup_time']:.2f} seconds
                - **Total Loading Time**: {model_load_stats['total_time']:.2f} seconds
                - **Compilation Disabled**: {model_load_stats['compilation_disabled']}
                """)
        
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
                
                # Advanced parameters in a collapsible section
                with gr.Accordion("Advanced Parameters", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.6,
                        step=0.05,
                        info="Higher values make output more random, lower values more deterministic"
                    )
                    
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        info="Nucleus sampling: only consider tokens with top_p cumulative probability"
                    )
                    
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=200,
                        maximum=2000,
                        value=1200,
                        step=100,
                        info="Maximum number of tokens to generate"
                    )
                    
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.3,
                        step=0.05,
                        info="Penalize repeated tokens. Higher values reduce repetition"
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
                
                with gr.Accordion("Generation Info", open=False):
                    gen_params_display = gr.JSON(label="Parameters Used")
        
        def process_text(text_input, voice, temperature, top_p, max_tokens, repetition_penalty):
            try:
                # Create a unique filename for each request to avoid conflicts
                unique_id = str(uuid.uuid4())[:8]
                temp_filename = os.path.join(output_dir, f"output_{unique_id}.wav")
                
                # Create parameters dictionary for display
                params = {
                    "voice": voice,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "text_length": len(text_input)
                }
                
                # Validate input
                if not text_input or len(text_input.strip()) == 0:
                    error_msg = "Error: Input text cannot be empty"
                    print(error_msg)
                    yield None, error_msg, error_msg, {"error": error_msg}
                    return
                
                # Update status
                status_message = "Processing request..."
                print(status_message)
                yield temp_filename, status_message, status_message, params
                
                # Generate the speech
                print(f"Sending generation request with params: {params}")
                wav_buffer, duration, gen_time = generate_speech(
                    prompt=text_input, 
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                )
                
                # Check if we got valid audio data
                if wav_buffer.getbuffer().nbytes == 0:
                    raise Exception("Generated audio buffer is empty")
                
                # Save to file for Gradio to display
                print(f"Saving audio to file: {temp_filename}")
                with open(temp_filename, "wb") as f:
                    f.write(wav_buffer.getvalue())
                
                # Verify the file was created and has content
                if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                    raise Exception(f"Failed to save audio file or file is empty: {temp_filename}")
                
                # Update parameters with results
                params["duration"] = f"{duration:.2f}s"
                params["generation_time"] = f"{gen_time:.2f}s"
                params["rtf"] = f"{gen_time/duration:.2f}x" if duration > 0 else "N/A"
                params["file_size"] = f"{os.path.getsize(temp_filename) / 1024:.2f} KB"
                
                # Return results
                result_message = f"Generation completed in {gen_time:.2f}s"
                print(result_message)
                yield temp_filename, f"{duration:.2f}", f"{gen_time:.2f}", params
                
            except Exception as e:
                import traceback
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                yield None, "Error", error_msg, {"error": str(e), "traceback": traceback.format_exc()}
        
        generate_btn.click(
            fn=process_text,
            inputs=[text_input, voice, temperature, top_p, max_tokens, repetition_penalty],
            outputs=[audio_output, duration_output, generation_time, gen_params_display],
            show_progress=True
        ).then(
            fn=lambda: "Ready",
            inputs=None,
            outputs=status_msg
        )
    
    # Add system info display
    with gr.Accordion("System Information", open=False):
        system_info = {
            "Python Version": sys.version,
            "Gradio Version": gr.__version__,
            "Platform": sys.platform,
            "Output Directory": os.path.abspath(output_dir)
        }
        gr.JSON(value=system_info, label="System Info")
    
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
    parser.add_argument('--skip-warmup', action='store_true',
                        help='Skip the warm-up generation during model loading')
    parser.add_argument('--fast-load', action='store_true',
                        help='Disable PyTorch compilation for faster loading (may affect inference speed)')
    args = parser.parse_args()
    
    # Set environment variables if fast-load is requested
    if args.fast_load:
        print("Fast loading enabled - PyTorch compilation will be disabled")
    
    # Load the model once at startup
    print("Initializing Orpheus TTS model...")
    global_model = load_model(args.model_path, disable_compile=args.fast_load)
    
    # Create and launch the interface
    print("Starting Gradio web interface...")
    app = create_interface(args.model_path)
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True
    ) 