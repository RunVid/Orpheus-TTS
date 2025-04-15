import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import os
import uuid
import gc
import torch
import time
import logging
import traceback
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
output_dir = "outputs"
request_counter = 0  # 记录第几次请求

os.makedirs(output_dir, exist_ok=True)

def log_memory_usage(tag=""):
    """打印 CUDA 显存占用信息，用于调试"""
    logger.info(f"=== MEMORY USAGE {tag} ===")
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"CUDA memory reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logger.info(f"CUDA max mem alloc:    {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    logger.info("=======================")

def load_model(model_path):
    """加载 Orpheus TTS 模型（同步）"""
    logger.info(f"Loading model from: {model_path}")
    log_memory_usage("BEFORE_MODEL_LOAD")
    t0 = time.time()

    model_instance = OrpheusModel(model_name=model_path)  # 这里不做任何异步操作

    elapsed = time.time() - t0
    logger.info(f"Model loaded in {elapsed:.2f} seconds")
    log_memory_usage("AFTER_MODEL_LOAD")
    return model_instance

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """手动创建 WAV 文件头（若需要实时流式输出可以用，但此处保留供参考）"""
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
    """最简同步生成：一次性拿到模型返回的所有音频，再写文件"""
    global model, request_counter
    request_counter += 1
    req_index = request_counter

    logger.info(f"=== GENERATE SPEECH REQUEST #{req_index} ===")
    logger.info(f"Voice: {voice}, Input text length: {len(prompt)}")

    log_memory_usage("BEFORE_GENERATION")

    start_t = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"Starting TTS with request_id: {request_id}")

    filename = os.path.join(output_dir, f"output_{uuid.uuid4()}.wav")
    logger.info(f"Will save WAV to: {filename}")

    try:
        # 直接调用模型
        logger.info("Calling model.generate_speech...")
        t_gen0 = time.time()

        # model.generate_speech 通常返回一个“音频块”迭代器，所以我们把它全部收集起来
        syn_tokens = list(model.generate_speech(prompt=prompt, voice=voice, request_id=request_id))
        logger.info(f"model.generate_speech returned in {time.time() - t_gen0:.2f}s")

        # 把所有音频块拼起来
        audio_buffer = b"".join(syn_tokens)

        logger.info("Writing WAV file (single pass)")
        t_wav0 = time.time()
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_buffer)
        logger.info(f"WAV writing done in {time.time() - t_wav0:.2f} s")

        # 日志
        elapsed_time = time.time() - start_t
        logger.info(f"Generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Audio length: {len(audio_buffer)} bytes")

        # 清理显存
        logger.info("Starting memory cleanup")
        log_memory_usage("BEFORE_CLEANUP")
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage("AFTER_CLEANUP")

        logger.info(f"Audio saved to: {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error during speech generation: {str(e)}")
        logger.error(traceback.format_exc())
        log_memory_usage("AFTER_ERROR")
        raise

def create_interface():
    """创建最简单的 Gradio 界面"""
    default_prompt = (
        "Man, the way social media has, um, completely changed how we interact is just wild, right? "
        "Like, we're all connected 24/7 but somehow people feel more alone than ever. "
        "And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."
    )

    with gr.Blocks(title="Simple Orpheus TTS") as demo:
        gr.Markdown("# Orpheus TTS Simple Interface")

        with gr.Row():
            with gr.Column():
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
                generate_btn = gr.Button("Generate Speech", variant="primary")

            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech", type="filepath")
                debug_info = gr.Textbox(label="Debug Info", value="", lines=2, interactive=False)

        # 点击按钮的处理函数
        def process(text, selected_voice):
            global request_counter
            logger.info("===== New Request =====")
            logger.info(f"Voice: {selected_voice}")

            if not text.strip():
                err_msg = "ERROR: Empty text"
                logger.error(err_msg)
                return None, err_msg

            req_str = f"Request #{request_counter+1}"
            try:
                log_memory_usage("BEFORE_PROCESSING")
                start_time = time.time()

                out_path = generate_speech(text, selected_voice)

                cost_time = time.time() - start_time
                log_memory_usage("AFTER_PROCESSING")
                msg = f"{req_str} done in {cost_time:.2f}s"
                logger.info(msg)
                return out_path, msg

            except Exception as exc:
                err = f"{req_str} | ERROR: {exc}"
                logger.error(err)
                return None, err

        generate_btn.click(
            fn=process,
            inputs=[text_input, voice],
            outputs=[audio_output, debug_info]
        )

    return demo

if __name__ == "__main__":
    model_path = "/home/ubuntu/models/orpheus-3b-0.1-ft"
    logger.info("====== STARTING ORPHEUS TTS SIMPLE ======")
    logger.info(f"Current working directory: {os.getcwd()}")

    try:
        logger.info("Initializing model...")
        model = load_model(model_path)
        logger.info("Model loaded successfully")

        log_memory_usage("INITIAL")

        logger.info("Starting web interface...")
        app = create_interface()
        # 不要开启多线程或asyncio.run等，直接同步方式启动
        app.launch(server_name="0.0.0.0", server_port=7860, share=True)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
