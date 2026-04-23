import os
import streamlit as st
from openai import OpenAI
from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import PyPDF2

# 🔑 Token
HF_TOKEN = os.getenv("")

if not HF_TOKEN:
    st.error("HF_TOKEN missing! Add in Railway variables.")
    st.stop()

# 🧠 AI client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# 🎨 UI config
st.set_page_config(page_title="Anshu Study AI 💖", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4);
    color: black;
}
h1 {
    text-align: center;
}
.stButton button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    width: 100%;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💖 Anshu Study AI</h1>", unsafe_allow_html=True)

# 📥 Inputs
image = st.file_uploader("📷 Upload Question Image", type=["png","jpg","jpeg"])
pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])

pdf_text = ""

# 📄 PDF read
if pdf:
    reader = PyPDF2.PdfReader(pdf)
    for page in reader.pages:
        pdf_text += page.extract_text()

# 📷 Image preview
if image:
    img = Image.open(image)
    st.image(img, caption="Uploaded Image")

    # OCR
    img_gray = np.array(img.convert("L"))
    text = pytesseract.image_to_string(img_gray)

    st.markdown("### 📄 Detected Question")
    st.code(text)

    if st.button("🚀 Solve Now"):

        prompt = f"""
        You are a BTech expert tutor.

        Solve step-by-step.
        Explain clearly in simple Hindi + English.
        Give final answer.
        Double check result.

        Question: {text}
        PDF Content: {pdf_text[:1500]}
        """

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct:together",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        st.markdown(f"""
        <div class="card">
        <h3>🤖 Solution</h3>
        <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)

        # 📊 Example diagram
        st.markdown("### 📊 Example Graph")
        x = np.linspace(0,10,50)
        y = x**2
        plt.figure()
        plt.plot(x,y)
        st.pyplot(plt)

# 💖 Footer
st.markdown("""
<hr>
<div style='text-align:center;'>
💖 Made for Anshu | Created by Akshay
</div>
""", unsafe_allow_html=True)