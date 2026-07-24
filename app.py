import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import ollama
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile


# ----------------------------
# Page Settings
# ----------------------------
st.set_page_config(
    page_title="Class 3 CBSE AI Chatbot",
    page_icon="📚",
    layout="centered"
)


# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    .stButton button {
        width:100%;
        border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Header
# ----------------------------
st.title("🏫 ABC School")

st.title("📚 Class 3 CBSE AI Chatbot")

st.success(
    "👋 Welcome!\n\n"
    "I am your Class 3 CBSE AI Learning Assistant.\n\n"
    "Ask questions from:\n\n"
    "📖 English\n"
    "➕ Maths\n"
    "🌍 EVS\n"
    "🔬 Science"
)


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:

    st.header("📚 Subjects")

    subject = st.selectbox(
        "Choose Subject",
        [
            "All Subjects",
            "📖 English",
            "➕ Maths",
            "🌍 EVS",
            "🔬 Science"
        ]
    )

    st.write("---")

    st.subheader("💡 Suggested Questions")

    st.write("• What is a noun?")
    st.write("• What is a verb?")
    st.write("• What is addition?")
    st.write("• Explain plants.")
    st.write("• What is the solar system?")

    st.write("---")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------
# Load Models
# ----------------------------
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


index = faiss.read_index(
    "class3_index.faiss"
)


with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)



# ----------------------------
# Chat History
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# ----------------------------
# Voice + Text Input
# ----------------------------

col1, col2 = st.columns([4,1])


with col1:

    question = st.chat_input(
        "Ask your question..."
    )


with col2:

    voice_question = speech_to_text(
        language="en",
        start_prompt="🎤 Speak",
        stop_prompt="⏹ Stop"
    )


if voice_question:
    question = voice_question



# ----------------------------
# Chat Processing
# ----------------------------

if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):
        st.markdown(question)



    embedding = model.encode(
        [question]
    )


    distance, result = index.search(
        embedding,
        3
    )


    if distance[0][0] > 1.0:

        answer = (
            "❌ Sorry! I can only answer questions "
            "related to the Class 3 CBSE syllabus."
        )


    else:

        context = ""

        for idx in result[0]:
            context += chunks[idx] + "\n\n"


        prompt = f"""
You are an experienced Class 3 CBSE teacher.

Rules:

1. Answer ONLY using the Context below.
2. Never make up information.
3. Keep answers simple.
4. Use easy English suitable for Class 3 students.
5. Give examples whenever possible.

Context:
{context}

Question:
{question}

Answer:
"""


        response = ollama.chat(

            model="qwen2.5:7b",

            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )


        answer = response["message"]["content"]



    # ----------------------------
    # Show Answer
    # ----------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)



        # 🔊 Text to Speech

        tts = gTTS(
            text=answer,
            lang="en"
        )


        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )


        tts.save(
            audio_file.name
        )


        st.audio(
            audio_file.name,
            format="audio/mp3"
        )



    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )