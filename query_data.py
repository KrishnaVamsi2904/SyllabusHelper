import streamlit as st
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_CHROMA_PATH = "chromadb_"
DEPARTMENTS = {
    "Computer Science and Engineering (CSE)": "cse",
    "Artificial Intelligence and Machine Learning (CS-AIML)": "aiml",
    "Electronics and Communication Engineering (ECE)": "ece",
    "Information Technology (IT)": "it",
    "Computer Science and Business System (CSBS)": "csbs",
    "Data Science (DS)": "ds"
}

PROMPT_TEMPLATE = """
You are a helpful assistant that helps students by giving answers to syllabus-related queries.
You are given a context below. Give an answer to the question below using only information from this context.

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str, chroma_path: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = genai.GenerativeModel('gemini-1.5-pro-002')
    response = model.generate_content(prompt)

    response_text = response.text
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    return formatted_response, response_text, sources, context_text

def main():
    st.set_page_config(page_title="SyllabusBuddy", page_icon="📘", layout="centered")
    st.markdown(
        """
        <style>
            /* Background color for the page */
            body {
                background-color: #f0f8ff;
            }
            /* Center the main content container */
            .main-container {
                max-width: 800px;
                margin: auto;
                padding: 2rem;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
            /* Style for main header */
            .main-header {
                color: #2c3e50;
                font-size: 2em;
                text-align: center;
                font-weight: bold;
            }
            /* Subheader styling */
            .subheader {
                color: #34495e;
                font-weight: bold;
                margin-top: 10px;
            }
            /* Styling for the answer text */
            .answer {
                color: #2e8b57;
                font-size: 1.2em;
                font-weight: bold;
            }
            /* Source text styling */
            .source {
                color: #8b0000;
            }
            /* Footer styling */
            .footer {
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                margin-top: 20px;
            }
            /* Style for the button */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<p class="main-header">Syllabus Helper📘</p>', unsafe_allow_html=True)
    st.write("Get syllabus-related answers specific to your department!")

    st.markdown('<p class="subheader">Select your department:</p>', unsafe_allow_html=True)
    department = st.selectbox("Department", ["Select a department"] + list(DEPARTMENTS.keys()), index=0)
    if department != "Select a department":
        chroma_path = f"{BASE_CHROMA_PATH}{DEPARTMENTS[department]}"

        st.markdown('<p class="subheader">Ask your question below:</p>', unsafe_allow_html=True)
        query_text = st.text_input("")
        if st.button("Submit:"):
            if query_text:
                st.write("Hold on!")
                formatted_response, response_text, sources, context_text = query_rag(query_text, chroma_path)

                st.markdown('<p class="answer">Answer:</p>', unsafe_allow_html=True)
                st.write(response_text)

                st.markdown('<p class="answer">Sources:</p>', unsafe_allow_html=True)
                st.write(sources)

                st.markdown('<p class="answer">Context:</p>', unsafe_allow_html=True)
                st.write(context_text)
            else:
                st.write("Please enter a query.")

    st.markdown('</div>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()