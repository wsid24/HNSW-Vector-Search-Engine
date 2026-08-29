import os
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def main():
    print("Loading AG News dataset...")
    # Load ag_news train split
    dataset = load_dataset("fancyzhx/ag_news", split="train")
    
    # Take first 100,000 entries
    num_samples = 100000
    texts = dataset["text"][:num_samples]
    
    print(f"Loaded {len(texts)} texts. Initializing model all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    batch_size = 256
    embeddings = []
    
    print(f"Starting encoding in batches of {batch_size}...")
    start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
        
        # Print progress every 10 batches
        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i // batch_size} / {len(texts) // batch_size} ({(i/len(texts))*100:.1f}%)")
            
    embeddings = np.vstack(embeddings)
    print(f"Encoding complete in {time.time() - start_time:.1f} seconds. Shape: {embeddings.shape}")
    
    # Split into database and queries
    db_embeddings = embeddings[:-1000]
    query_embeddings = embeddings[-1000:]
    query_texts = texts[-1000:]
    
    os.makedirs("../data", exist_ok=True)
    
    print("Saving database embeddings...")
    db_embeddings.astype(np.float32).tofile("../data/embeddings.bin")
    
    print("Saving query embeddings...")
    query_embeddings.astype(np.float32).tofile("../data/queries.bin")
    
    print("Saving query texts...")
    with open("../data/queries.txt", "w", encoding="utf-8") as f:
        for text in query_texts:
            f.write(text.replace('\n', ' ') + "\n")
            
    print("Done!")
    print(f"data/embeddings.bin size: {os.path.getsize('../data/embeddings.bin')} bytes")
    print(f"data/queries.bin size: {os.path.getsize('../data/queries.bin')} bytes")
    print(f"Database shape: {db_embeddings.shape}")
    print(f"Query shape: {query_embeddings.shape}")

if __name__ == "__main__":
    main()
