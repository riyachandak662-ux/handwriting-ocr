"""Handwriting to digital text web app (Google Gemini + Gradio), ready for Render.

Environment variable needed: GEMINI_API_KEY
Start command on Render:     python app.py
"""
import io
import os

import gradio as gr
from google import genai
from google.genai import types
from PIL import Image, ImageOps

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),  # give up after 60 seconds
)

# Tried in order; if one is unavailable or rate-limited, the next is used.
# Free-tier model names change over time. Edit this list if you get "model not found".
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

PROMPT = (
    'Transcribe all handwriting in this image exactly as written, keeping the original line breaks. '
    'Do not correct spelling, grammar, or math mistakes. '
    'Write math in plain readable text, NOT LaTeX: use ^ for powers (x^2), / for fractions (1760/x^3), '
    'write combinations like 12C9, and use Unicode symbols like × ÷ √ π. '
    'No dollar signs and no backslash commands. '
    'Mark any word or symbol you cannot read as [?]. '
    'Output only the transcription, with no commentary.'
)

LANGUAGES = ["Auto-detect", "English", "Hindi", "Marathi", "Gujarati",
             "Tamil", "Telugu", "Bengali", "Urdu", "French", "Spanish", "German"]


def prepare_image(img: Image.Image) -> bytes:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((2000, 2000))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def transcribe(img: Image.Image, language: str) -> str:
    if img is None:
        return "Please upload or capture a photo first."
    prompt = PROMPT
    if language != "Auto-detect":
        prompt += f" The handwriting is in {language}."
    image_part = types.Part.from_bytes(data=prepare_image(img), mime_type="image/jpeg")
    errors = []
    for model in MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(temperature=0),
            )
            if resp.text:
                return resp.text.strip()
            errors.append(f"{model}: empty response")
        except Exception as e:
            errors.append(f"{model}: {e}")
    return "Error: all models failed -> " + " | ".join(errors)


demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Image(type="pil", label="Photo of handwriting", sources=["upload", "webcam"]),
        gr.Dropdown(LANGUAGES, value="Auto-detect", label="Language"),
    ],
    outputs=gr.Textbox(lines=14, label="Digital text", show_copy_button=True),
    title="Handwriting to Text",
    description="Upload a clear photo of handwriting and get editable digital text.",
    flagging_mode="never",
)

if __name__ == "__main__":
    # Render tells the app which port to use through the PORT variable.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
