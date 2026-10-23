
import streamlit as st
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄"
)

st.title("📄 AI PDF Question Answering Assistant")
st.write("Upload a PDF and ask questions about its content.")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_chunks(question, chunks, index, top_k=3):
    question_embedding = model.encode([question])
    question_embedding = question_embedding.astype("float32")

    k = min(top_k, len(chunks))
    distances, indices = index.search(question_embedding, k)

    results = []
    for i in indices[0]:
        if i >= 0:
            results.append(chunks[i])

    return results


uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    full_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        full_text += f"\n[Page {page_number}]\n{page_text}"

    if not full_text.strip():
        st.warning(
            "No readable text found. Please use a PDF with selectable text."
        )
    else:
        st.success(
            f"PDF uploaded successfully! Pages: {len(reader.pages)}"
        )

        chunks = split_text(full_text)
        st.write(f"PDF divided into {len(chunks)} text chunks.")

        if chunks:
            index = create_faiss_index(chunks)
            st.success("PDF search index created successfully!")

            with st.expander("View extracted PDF text"):
                st.text(full_text[:10000])

            question = st.text_input(
                "Ask a question about your PDF"
            )

            if st.button("Find answer"):
                if not question.strip():
                    st.warning("Please enter a question.")

                elif not api_key:
                    st.error(
                        "Groq API key is missing. Check your .env file."
                    )

                else:
                    try:
                        relevant_chunks = search_chunks(
                            question, chunks, index, top_k=3
                        )
                        context = "\n\n".join(relevant_chunks)

                        with st.spinner("AI is finding the answer..."):
                            client = Groq(api_key=api_key)

                            response = client.chat.completions.create(
                                model="openai/gpt-oss-20b",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "Answer using only the supplied "
                                            "PDF text. If the answer is missing, "
                                            "say: 'The answer was not found "
                                            "in this PDF.'"
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            
"content": (
    "You are an assistant that answers questions using the supplied "
    "PDF excerpts only. "
    "Give a clear, simple, and accurate answer. "
    "Do not use outside knowledge. "
    "If the excerpts do not contain enough information, say: "
    "'The answer was not found in the retrieved PDF text.' "
    "Do not invent facts. "
    "If useful, organize the answer using short bullet points."
)

                                            f"Relevant PDF text:\n{context}\n\n"
                                            f"Question: {question}"
                                        )
                                    }
                                ]
                            )

                        st.subheader("AI Answer")
                        st.write(
                            response.choices[0].message.content
                        )

                    except Exception as e:
                        st.error(f"AI request failed: {e}")
