from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Step 1: Read knowledge file
with open("class3_cbse.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Split text into small chunks
chunks = text.split("###")

print("Knowledge chunks:")
for chunk in chunks:
    print("----------------")
    print(chunk)

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks into embeddings
embeddings = model.encode(chunks)

# Step 5: Create FAISS vector database
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Step 6: Add embeddings to FAISS
index.add(embeddings)

# Step 7: Save FAISS database
faiss.write_index(index, "class3_index.faiss")

# Save text chunks also
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Vector database created successfully!")