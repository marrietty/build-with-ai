import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import chromadb
from chromadb.utils import embedding_functions
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Show title and description.
st.set_page_config(page_title="AI Resume Screener Chatbot", layout="wide")
st.title("🤖 Resume Screener")
st.write(
    "This chatbot helps HR teams analyze resumes by summarizing content, checking for qualification matches, and providing critiques. "
    "It uses Google Gemini and ChromaDB to evaluate and store resumes."
)

# Ask user for their Google API key
api_key = st.text_input("🔑 Google AI API Key", type="password")
if not api_key:
    st.info("Please add your Google AI API key to continue.", icon="🗝️")
    st.stop()

# Initialize Google AI
genai.configure(api_key=api_key)
gemini = GenerativeModel("gemini-2.0-flash")

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("resumes")

# Extract text from uploaded PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Function to analyze resume with Gemini
def analyze_resume(resume_text, job_description):
    prompt = f"""
    You are an HR Assistant.
    Analyze the following resume and:
    1. Summarize the Skills, Experience, and Education.
    2. Compare it to the following job requirement: {job_description}
    3. Provide a critique with strengths and suggestions for improvement.

    Resume:
    {resume_text}
    """
    response = gemini.generate_content(prompt)
    return response.text

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload job description and resume
st.subheader("📄 Job Description")
job_description = st.text_area("Paste the job requirements here:", height=150)

st.subheader("📤 Upload Resume")
uploaded_file = st.file_uploader("Upload a resume (PDF only):", type=["pdf"])

# If input provided, analyze resume
if uploaded_file and job_description:
    with st.spinner("Analyzing Resume with Gemini AI..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        analysis = analyze_resume(resume_text, job_description)

        # Store interaction
        st.session_state.messages.append({"role": "user", "content": resume_text})
        st.session_state.messages.append({"role": "assistant", "content": analysis})

        # Display chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Save to ChromaDB
        collection.add(
            documents=[resume_text],
            metadatas=[{"filename": uploaded_file.name, "job_desc": job_description, "analysis": analysis}],
            ids=[uploaded_file.name]
        )

        st.success("Resume has been analyzed and stored!")