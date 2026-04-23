import os
import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
import PyPDF2

# 🔑 Token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN missing! Check Railway Variables")
    st.stop()

try:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
except Exception as e:
    st.error(f"Client Error: {e}")
    st.stop()

st.title("💖 Anshu Study AI")

# OCR
def extract_text(image_file):
    try:
        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": image_file},
            data={"apikey": "helloworld"}
        )
        result = response.json()
        return result["ParsedResults"][0]["ParsedText"]
    except:
        return "OCR failed"

image = st.file_uploader("Upload Image")

if image:
    st.image(image)

    if st.button("Solve"):
        text = extract_text(image)
        st.write("Question:", text)

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct:together",
                messages=[{"role": "user", "content": text}],
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI Error: {e}")
