import gradio as gr
from orpheus_tts import OrpheusModel
import wave
import os

# 全局变量 - 只保留模型
model = None
output_dir = "outputs"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def load_model(model_path):
    """加载Orpheus TTS模型 - 最简单实现"""
    print(f"正在加载模型: {model_path}")
    return OrpheusModel(model_name=model_path)

def generate_speech(prompt, voice="tara"):
    """生成语音 - 最简单实现"""
    global model
    
    print(f"生成语音: {voice}")
    print(f"输入文本: {prompt}")
    
    # 简单直接地生成语音
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice
    )
    
    # 生成唯一文件名
    import uuid
    filename = os.path.join(output_dir, f"output_{uuid.uuid4()}.wav")
    
    # 保存为WAV文件
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        
        for audio_chunk in syn_tokens:
            wf.writeframes(audio_chunk)
    
    return filename

def create_interface():
    """创建最简单的Gradio界面"""
    # 默认文本
    default_prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''
    
    # 创建界面
    with gr.Blocks(title="Simple Orpheus TTS") as demo:
        gr.Markdown("# Orpheus TTS 简易界面")
        
        with gr.Row():
            with gr.Column():
                # 输入
                text_input = gr.Textbox(
                    label="文本输入",
                    value=default_prompt,
                    lines=5
                )
                
                voice = gr.Dropdown(
                    label="语音",
                    choices=["tara", "emma", "bella", "antoni", "josh", "michael"],
                    value="tara"
                )
                
                # 生成按钮
                generate_btn = gr.Button("生成语音", variant="primary")
                
            with gr.Column():
                # 输出
                audio_output = gr.Audio(
                    label="生成的语音",
                    type="filepath"
                )
        
        # 处理函数
        def process(text, voice):
            if not text.strip():
                return None
            
            try:
                return generate_speech(text, voice)
            except Exception as e:
                print(f"错误: {str(e)}")
                return None
        
        # 事件绑定
        generate_btn.click(
            fn=process,
            inputs=[text_input, voice],
            outputs=audio_output
        )
    
    return demo

if __name__ == "__main__":
    # 加载模型 - 使用默认路径
    model_path = "/home/ubuntu/models/orpheus-3b-0.1-ft"
    print("初始化Orpheus TTS模型...")
    model = load_model(model_path)
    
    # 创建并启动界面
    print("启动Web界面...")
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 