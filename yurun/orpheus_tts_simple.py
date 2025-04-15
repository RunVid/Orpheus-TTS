import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import os
import uuid
import gc  # Added garbage collection
import torch  # Added torch import
import time  # Added for performance logging
import logging  # Added for better logging
import traceback  # Added for detailed error tracking
import struct  # Added for WAV header
import threading  # Added for thread information

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
model = None
output_dir = "outputs"
request_counter = 0  # Track number of requests

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def log_memory_usage(tag=""):
    """Log detailed memory usage"""
    logger.info(f"=== MEMORY USAGE {tag} ===")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logger.info(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    logger.info("=======================")

def load_model(model_path):
    """Load Orpheus TTS model - simplest implementation"""
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Current thread: {threading.current_thread().name}")
    log_memory_usage("BEFORE_MODEL_LOAD")
    
    model_start_time = time.time()
    model_instance = OrpheusModel(model_name=model_path)
    model_load_time = time.time() - model_start_time
    
    logger.info(f"Model loaded in {model_load_time:.2f} seconds")
    log_memory_usage("AFTER_MODEL_LOAD")
    
    return model_instance

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV header for streaming"""
    logger.info("Creating WAV header")
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,       
        b'WAVE',
        b'fmt ',
        16,                  
        1,             
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

def generate_speech(prompt, voice="tara"):
    """Generate speech - simplest implementation"""
    global model, request_counter
    request_counter += 1
    
    logger.info(f"=== GENERATE SPEECH REQUEST #{request_counter} ===")
    logger.info(f"Thread: {threading.current_thread().name}")
    logger.info(f"Generate speech called with voice: {voice}")
    logger.info(f"Input text length: {len(prompt)} characters")
    
    log_memory_usage("BEFORE_GENERATION")
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"Starting generation with request_id: {request_id}")
    
    # Generate unique filename
    filename = os.path.join(output_dir, f"output_{uuid.uuid4()}.wav")
    logger.info(f"Will save to file: {filename}")
    
    try:
        # Check model state before generation
        logger.info(f"Model object type: {type(model)}")
        logger.info(f"Model object ID: {id(model)}")
        
        # Generate speech directly
        logger.info("Calling model.generate_speech")
        gen_start_time = time.time()
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice=voice,
            request_id=request_id
        )
        logger.info(f"model.generate_speech returned in {time.time() - gen_start_time:.2f} seconds")
        
        # Save to WAV file - streaming mode
        logger.info("Opening WAV file for writing")
        wav_start_time = time.time()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            logger.info("Starting to process audio chunks")
            chunk_count = 0
            chunk_sizes = []
            
            for audio_chunk in syn_tokens:
                chunk_count += 1
                chunk_sizes.append(len(audio_chunk))
                
                if chunk_count % 5 == 0:
                    logger.info(f"Processing chunk #{chunk_count}, size: {len(audio_chunk)} bytes")
                
                wf.writeframes(audio_chunk)
        
        wav_time = time.time() - wav_start_time
        logger.info(f"WAV writing completed in {wav_time:.2f} seconds")
        
        # Log chunk statistics
        if chunk_sizes:
            logger.info(f"Min chunk size: {min(chunk_sizes)} bytes")
            logger.info(f"Max chunk size: {max(chunk_sizes)} bytes")
            logger.info(f"Avg chunk size: {sum(chunk_sizes)/len(chunk_sizes):.2f} bytes")
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total chunks processed: {chunk_count}")
        
        # Force a cleanup after generation
        logger.info("Starting memory cleanup")
        log_memory_usage("BEFORE_CLEANUP")
        
        cleanup_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        cleanup_time = time.time() - cleanup_start
        
        log_memory_usage("AFTER_CLEANUP")
        logger.info(f"Cleanup completed in {cleanup_time:.2f} seconds")
        
        logger.info(f"Audio saved to: {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error during speech generation: {str(e)}")
        logger.error(traceback.format_exc())
        log_memory_usage("AFTER_ERROR")
        raise

def create_interface():
    """Create the simplest Gradio interface"""
    # Default text
    default_prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
    
    # Create interface
    with gr.Blocks(title="Simple Orpheus TTS") as demo:
        gr.Markdown("# Orpheus TTS Simple Interface")
        
        with gr.Row():
            with gr.Column():
                # Input
                text_input = gr.Textbox(
                    label="Text Input",
                    value=default_prompt,
                    lines=5
                )
                
                voice = gr.Dropdown(
                    label="Voice",
                    choices=["tara", "emma", "bella", "antoni", "josh", "michael"],
                    value="tara"
                )
                
                # Generate button
                generate_btn = gr.Button("Generate Speech", variant="primary")
                
            with gr.Column():
                # Output
                audio_output = gr.Audio(
                    label="Generated Speech",
                    type="filepath"
                )
                
                # Add debug info display
                debug_info = gr.Textbox(
                    label="Debug Info",
                    value="",
                    lines=2,
                    interactive=False
                )
        
        # Processing function
        def process(text, voice):
            global request_counter
            
            process_id = f"REQUEST-{request_counter+1}"
            logger.info(f"============= {process_id} =============")
            logger.info(f"Process function called for voice: {voice}")
            logger.info(f"Button clicked at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Thread ID: {threading.get_ident()}")
            logger.info(f"Thread name: {threading.current_thread().name}")
            
            # Capture state before processing
            debug_text = f"Request #{request_counter+1} | Thread: {threading.current_thread().name}"
            
            if not text.strip():
                logger.error("Error: Empty text input")
                return None, debug_text + " | ERROR: Empty text"
            
            try:
                # Log memory state before processing
                log_memory_usage(f"{process_id}_BEFORE_PROCESSING")
                
                logger.info("Calling generate_speech function")
                process_start = time.time()
                result = generate_speech(text, voice)
                process_time = time.time() - process_start
                
                logger.info(f"Process function completed in {process_time:.2f} seconds, returning: {result}")
                
                # Log memory state after processing
                log_memory_usage(f"{process_id}_AFTER_PROCESSING")
                
                # Update debug info
                debug_text += f" | Success in {process_time:.2f}s"
                
                return result, debug_text
            except Exception as e:
                logger.error(f"Error in process function: {str(e)}")
                logger.error(traceback.format_exc())
                return None, debug_text + f" | ERROR: {str(e)}"
        
        # Event binding
        generate_btn.click(
            fn=process,
            inputs=[text_input, voice],
            outputs=[audio_output, debug_info]
        )
    
    return demo

if __name__ == "__main__":
    # Load model - use default path
    model_path = "/home/ubuntu/models/orpheus-3b-0.1-ft"
    logger.info("====== STARTING ORPHEUS TTS SIMPLE ======")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        logger.info("Initializing model...")
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Log initial memory state
        log_memory_usage("INITIAL")
        
        # Create and launch interface
        logger.info("Starting web interface...")
        app = create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc()) 