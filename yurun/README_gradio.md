# Orpheus TTS Web Interface

This is a web interface for Orpheus TTS using Gradio.

## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

## Running the Web Interface

Run the following command:

```
python yurun/orpheus_tts_gradio.py
```

This will:
1. Load the model from the default path ("/home/ubuntu/models/orpheus-3b-0.1-ft")
2. Start a local Gradio server and provide you with a URL to access the web interface

To specify a custom model path:

```
python yurun/orpheus_tts_gradio.py --model_path "/path/to/your/model"
```

## Features

- The model is loaded once at startup for better performance
- Enter text to convert to speech
- Select from various voices
- Generate and play speech directly in the browser
- Download the generated audio file
- View audio duration and generation time information

## Usage

1. Wait for the model to fully load (this may take some time)
2. Type or paste the text you want to convert to speech
3. Select a voice from the dropdown menu
4. Click the "Generate Speech" button
5. Once generated, you can play the audio directly in the browser or download it

The interface will also display the duration of the generated audio and how long it took to generate. 