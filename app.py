import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import google.generativeai as genai

# --------------------------
# Page Settings
# --------------------------

st.set_page_config(
    page_title="Class 3 CBSE AI Chatbot",
    page_icon="📚",
    layout="centered"
)

# --------------------------
# Header
# --------------------------

st.title("🏫 ABC School")
st.header("📚 Class 3 CBSE AI Chatbot")

st.write(
    """
👋 Welcome!

I am your Class 3 CBSE AI Learning Assistant.

Ask questions from:

📖 English  
➕ Maths  
🌍 EVS  
🔬 Science
"""
)

# --------------------------
# Sidebar
# --------------------------

with st.sidebar:

    st.title("📚 Subjects")

    subject = st.selectbox(
        "Choose Subject:",
        [
            "All Subjects",
            "English",
            "Maths",
            "EVS",
            "Science"
        ]
    )

    st.subheader("💡 Suggested Questions")

    st.write("• What is a noun?")
    st.write("• What is a verb?")
    st.write("• What is addition?")
    st.write("• Explain plants.")
    st.write("• What is the solar system?")

    if st.button("🧹 Clear Chat"):

        st.session_state.messages = []
        st.rerun()


# --------------------------
# Load Embedding Model
# --------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# --------------------------
# Load FAISS Database
# --------------------------

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "class3_index.faiss"
    )

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


index, chunks = load_database()


# --------------------------
# Gemini Configuration
# --------------------------

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)


# --------------------------
# Chat History
# --------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# --------------------------
# Chat Input
# --------------------------

question = st.chat_input(
    "Ask your question..."
)


if question:


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)



    # Create embedding

    embedding = model.encode(
        [question]
    )


    # Search FAISS

    distance, result = index.search(
        embedding,
        1
    )


    # Context checking

    if distance[0][0] > 1.0:


        answer = (
            "❌ Sorry! I can only answer "
            "questions related to the Class 3 CBSE syllabus."
        )


    else:


        context = chunks[result[0][0]]


        prompt = f"""

You are an experienced Class 3 CBSE teacher.

Rules:

1. Answer ONLY using the Context below.
2. Never make up information.
3. If the Context does not contain the answer, reply:
Sorry! I can only answer questions related to the Class 3 CBSE syllabus.
4. Keep the answer simple.
5. Use easy English suitable for a Class 3 student.
6. Give examples whenever possible.
7. Answer in 3-5 short sentences.

Context:

{context}


Question:

{question}


Answer:

"""


        response = gemini_model.generate_content(
            prompt
        )


        answer = response.text



    # Display Answer

    with st.chat_message("assistant"):

        st.markdown(answer)



    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )