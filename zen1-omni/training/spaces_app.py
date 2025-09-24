"""
Zen1-Omni HuggingFace Spaces Demo
Interactive multimodal AI with text, audio, image, and video understanding
"""

import os
import gradio as gr
import torch
import numpy as np
import soundfile as sf
from PIL import Image
import cv2
from typing import Optional, Tuple, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from qwen_omni_utils import process_mm_info
import spaces


# Configuration
MODEL_ID = os.getenv("MODEL_ID", "zen-ai/zen1-omni-30b-gspo")
MAX_LENGTH = 8192
MAX_NEW_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model storage
model = None
tokenizer = None


@spaces.GPU(duration=120)  # Request GPU for 2 minutes
def load_model():
    """Load Zen1-Omni model with optimizations"""
    global model, tokenizer

    if model is None:
        print(f"Loading Zen1-Omni model: {MODEL_ID}")

        # Quantization for Spaces
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )

        print("✓ Zen1-Omni model loaded")

    return model, tokenizer


@spaces.GPU(duration=60)
def generate_multimodal(
    text_input: str,
    audio_input: Optional[Tuple] = None,
    image_input: Optional[Image.Image] = None,
    video_input: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    generate_speech: bool = False,
    thinking_mode: bool = False
) -> Tuple[str, Optional[Tuple]]:
    """
    Generate response from Zen1-Omni with multimodal inputs
    """

    # Load model
    model, tokenizer = load_model()

    # Prepare conversation
    conversation = [
        {
            "role": "system",
            "content": "You are Zen1-Omni, an advanced multimodal AI assistant capable of understanding text, audio, images, and video."
        }
    ]

    # Build user message with multimodal content
    user_content = []

    if text_input:
        user_content.append({"type": "text", "text": text_input})

    if audio_input:
        # Save audio temporarily
        audio_path = "/tmp/input_audio.wav"
        sf.write(audio_path, audio_input[1], audio_input[0])
        user_content.append({"type": "audio", "audio": audio_path})

    if image_input:
        # Save image temporarily
        image_path = "/tmp/input_image.jpg"
        image_input.save(image_path)
        user_content.append({"type": "image", "image": image_path})

    if video_input:
        user_content.append({"type": "video", "video": video_input})

    conversation.append({"role": "user", "content": user_content})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    # Process multimodal data
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=True
    )

    # Tokenize and prepare inputs
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Add multimodal data if present
    if audios:
        inputs["audio"] = audios
    if images:
        inputs["images"] = images
    if videos:
        inputs["videos"] = videos

    inputs = inputs.to(model.device)

    # Generate response
    with torch.no_grad():
        if generate_speech:
            # Generate both text and audio
            text_ids, audio_output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                speaker="zen1",  # Zen1 voice
                thinker_return_dict_in_generate=thinking_mode,
                use_audio_in_video=True
            )

            # Decode text
            text_output = tokenizer.decode(
                text_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            # Process audio
            if audio_output is not None:
                audio_array = audio_output.reshape(-1).cpu().numpy()
                audio_tuple = (24000, audio_array)  # 24kHz sample rate
            else:
                audio_tuple = None

        else:
            # Text-only generation
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )

            text_output = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            audio_tuple = None

    return text_output, audio_tuple


def create_demo():
    """Create Gradio interface for Zen1-Omni"""

    with gr.Blocks(
        title="Zen1-Omni Multimodal Demo",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🌟 Zen1-Omni Multimodal AI

        Advanced AI assistant with multimodal understanding and generation capabilities.

        **Features:**
        - 🎯 Understands text, audio, images, and video
        - 🗣️ Generates both text and natural speech
        - 🧠 Chain-of-thought reasoning mode
        - 🌍 Supports 119 text languages and 19 audio languages
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Type your message or question...",
                    lines=3
                )

                audio_input = gr.Audio(
                    label="Audio Input (Optional)",
                    type="numpy",
                    source="microphone"
                )

                image_input = gr.Image(
                    label="Image Input (Optional)",
                    type="pil"
                )

                video_input = gr.Video(
                    label="Video Input (Optional)"
                )

                # Settings
                with gr.Accordion("Advanced Settings", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1
                    )

                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=128,
                        maximum=2048,
                        value=1024,
                        step=128
                    )

                    generate_speech = gr.Checkbox(
                        label="Generate Speech Output",
                        value=False
                    )

                    thinking_mode = gr.Checkbox(
                        label="Enable Chain-of-Thought",
                        value=False
                    )

                # Submit button
                submit_btn = gr.Button(
                    "🚀 Generate Response",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                # Output components
                text_output = gr.Textbox(
                    label="Zen1-Omni Response",
                    lines=10,
                    interactive=False
                )

                audio_output = gr.Audio(
                    label="Speech Output",
                    type="numpy",
                    visible=False
                )

                # Examples
                gr.Examples(
                    examples=[
                        ["What's the meaning of life?", None, None, None],
                        ["Describe this image in detail", None, "examples/sample.jpg", None],
                        ["Transcribe and translate this audio", "examples/speech.wav", None, None],
                        ["Summarize this video", None, None, "examples/video.mp4"]
                    ],
                    inputs=[text_input, audio_input, image_input, video_input],
                    label="Example Inputs"
                )

        # Event handlers
        def on_generate(*args):
            text_out, audio_out = generate_multimodal(*args)

            # Show/hide audio output
            audio_visible = audio_out is not None

            return text_out, audio_out, gr.update(visible=audio_visible)

        submit_btn.click(
            fn=on_generate,
            inputs=[
                text_input,
                audio_input,
                image_input,
                video_input,
                temperature,
                max_tokens,
                generate_speech,
                thinking_mode
            ],
            outputs=[text_output, audio_output, audio_output]
        )

        # Footer
        gr.Markdown("""
        ---
        **Model**: [Zen1-Omni-30B-GSPO](https://huggingface.co/zen-ai/zen1-omni-30b-gspo)
        | **Paper**: [arXiv](https://arxiv.org/abs/zen1-omni)
        | **GitHub**: [zen-ai/zen1-omni](https://github.com/zen-ai/zen1-omni)
        """)

    return demo


if __name__ == "__main__":
    # Load model on startup
    print("Initializing Zen1-Omni...")
    load_model()

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )