import os
import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
import PyPDF2

# 🔑 Token (Railway se aayega)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN missing! Add it in Railway Variables")
    st.stop()

# 🧠 AI client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# 🎨 UI
st.set_page_config(page_title="💖 Anshu Study AI", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ff9a9e, #fad0c4);
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
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💖 Anshu Study AI</h1>", unsafe_allow_html=True)

# 📷 OCR function (Railway compatible)
def extract_text(image_file):
    url = "https://api.ocr.space/parse/image"
    response = requests.post(
        url,
        files={"file": image_file},
        data={"apikey": "helloworld"}
    )
    result = response.json()
    try:
        return result["ParsedResults"][0]["ParsedText"]
    except:
        return "Text not detected properly"

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

    if st.button("🚀 Solve Now"):

        # 🧠 OCR
        text = extract_text(image)

        st.markdown("### 📄 Detected Question")
        st.code(text)

        # 🤖 AI solve
        prompt = f"""
        You are a BTech expert tutor.

        Solve step-by-step.
        Explain clearly in simple Hindi + English.
        Give final answer.
        Double check answer.

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

# 💖 Footer
st.markdown("""
<hr>
<div style='text-align:center;'>
💖 Made for Anshu | Created by Akshay
</div>
""", unsafe_allow_html=True)
