from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load FAISS vector database
index = faiss.read_index("class3_index.faiss")

# Load knowledge chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Class 3 CBSE Chatbot Ready! Type 'exit' to stop.")

while True:
    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert user question into vector
    question_embedding = model.encode([question])

    # Search in FAISS database
    distance, result = index.search(question_embedding, k=1)

    # Print the distance
    print("Distance:", distance[0][0])

    # Check if the question is related
    if distance[0][0] > 1.0:
        print("\nSorry! I can only answer questions related to the Class 3 CBSE syllabus.")
    else:
        answer = chunks[result[0][0]]

        print("\nAnswer:")
        print(answer)