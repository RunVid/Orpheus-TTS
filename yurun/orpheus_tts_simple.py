import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import os
import uuid

# Global variable - only store the model
model = None
output_dir = "outputs"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def load_model(model_path):
    """Load Orpheus TTS model - simplest implementation"""
    print(f"Loading model: {model_path}")
    return OrpheusModel(model_name=model_path)

def generate_speech(prompt, voice="tara"):
    """Generate speech - simplest implementation"""
    global model
    
    print(f"Generating speech with voice: {voice}")
    print(f"Input text: {prompt}")
    
    # Generate speech directly
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice,
        request_id=str(uuid.uuid4())
    )
    
    # Generate unique filename
    filename = os.path.join(output_dir, f"output_{uuid.uuid4()}.wav")
    
    # Save to WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        for audio_chunk in syn_tokens:
            wf.writeframes(audio_chunk)
    
    print(f"Audio saved to: {filename}")
    return filename

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
        
        # Processing function
        def process(text, voice):
            if not text.strip():
                print("Error: Empty text input")
                return None
            
            try:
                return generate_speech(text, voice)
            except Exception as e:
                print(f"Error: {str(e)}")
                return None
        
        # Event binding
        generate_btn.click(
            fn=process,
            inputs=[text_input, voice],
            outputs=audio_output
        )
    
    return demo

if __name__ == "__main__":
    # Load model - use default path
    model_path = "/home/ubuntu/models/orpheus-3b-0.1-ft"
    print("Initializing Orpheus TTS model...")
    model = load_model(model_path)
    
    # Create and launch interface
    print("Starting web interface...")
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 