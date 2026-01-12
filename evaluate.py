import os
import sys

# Ensure src is in path
sys.path.append(os.path.abspath("src"))

from retriever import Retriever
from PIL import Image
import matplotlib.pyplot as plt

def evaluate():
    retriever = Retriever(index_path="data/index.faiss", metadata_path="data/metadata.json")
    
    queries = {
        "Attribute Specific": "A person in a bright yellow raincoat.",
        "Contextual/Place": "Professional business attire inside a modern office.",
        "Complex Semantic": "Someone wearing a blue shirt sitting on a park bench.",
        "Style Inference": "Casual weekend outfit for a city walk.",
        "Compositional": "A red tie and a white shirt in a formal setting."
    }
    
    print(f"{'Query Type':<20} | {'Query':<50} | {'Top Match Score':<10}")
    print("-" * 90)
    
    for q_type, query in queries.items():
        results = retriever.search(query, k=3, compositional=True)
        print(f"\nQUERY: {query} ({q_type})")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['filename']} (Score: {res['score']:.4f})")
            
        # Optional: Save/Show images (commented out for headless run)
        # if results:
        #     img = Image.open(results[0]['path'])
        #     img.show()

if __name__ == "__main__":
    evaluate()
