import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from google import genai


# --------------------------
# Page Configuration
# --------------------------

st.set_page_config(
    page_title="Class 3 CBSE AI Chatbot",
    page_icon="📚"
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

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()



# --------------------------
# Load FAISS Database
# --------------------------

@st.cache_resource
def load_database():

    index = faiss.read_index(
        "class3_index.faiss"
    )


    with open(
        "chunks.pkl",
        "rb"
    ) as f:

        chunks = pickle.load(f)


    return index, chunks



index, chunks = load_database()



# --------------------------
# Gemini API
# --------------------------

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)



# --------------------------
# Chat Memory
# --------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(msg["content"])




# --------------------------
# User Question
# --------------------------

question = st.chat_input(
    "Ask your question..."
)


if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.write(question)



    # ----------------------
    # Create Embedding
    # ----------------------

    embedding = embedding_model.encode(
        [question]
    )


    # ----------------------
    # FAISS Search
    # ----------------------

    distance, result = index.search(
        embedding,
        1
    )



    if distance[0][0] > 1.0:


        answer = (
            "❌ Sorry! I can only answer "
            "questions related to the Class 3 CBSE syllabus."
        )


    else:


        context = chunks[result[0][0]]



        prompt = f"""

You are a Class 3 CBSE teacher.

Answer only from the context.

Rules:
- Use simple English.
- Give examples.
- Do not answer outside Class 3 CBSE syllabus.
- If answer is not available say:
Sorry! I can only answer questions related to the Class 3 CBSE syllabus.

Context:

{context}


Question:

{question}


Answer:

"""


        # ----------------------
        # Gemini Response
        # ----------------------

        try:

            response = client.models.generate_content(

                model="gemini-flash-latest",

                contents=prompt

            )


            answer = response.text


        except Exception as e:

            answer = f"""
⚠️ Gemini Error Details:

{e}
"""



    with st.chat_message("assistant"):

        st.write(answer)



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

