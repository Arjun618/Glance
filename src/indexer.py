import os
os.environ["USE_TF"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import torch
import faiss
import numpy as np
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tqdm

class Indexer:
    def __init__(self, index_path="data/index.faiss", metadata_path="data/metadata.json", model_name="patrickjohncyh/fashion-clip"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        print(f"Loading Model: {model_name}...")
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}, falling back to openai/clip-vit-base-patch32. Error: {e}")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
        self.index = None
        self.metadata = []

    def process_images(self, image_dir):
        image_paths = glob.glob(os.path.join(image_dir, "*"))
        valid_image_paths = []
        
        print("Filtering valid images...")
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for p in image_paths:
            if os.path.splitext(p)[1].lower() in extensions:
                valid_image_paths.append(p)
        
        print(f"Found {len(valid_image_paths)} valid images.")
        
        if not valid_image_paths:
            return

        batch_size = 32
        embeddings_list = []
        
        print("Generating embeddings...")
        for i in tqdm.tqdm(range(0, len(valid_image_paths), batch_size)):
            batch_paths = valid_image_paths[i:i+batch_size]
            
            try:
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                
                batch_embeddings = outputs.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                embeddings_list.append(batch_embeddings)
                
                for p in batch_paths:
                    self.metadata.append({"path": os.path.abspath(p), "filename": os.path.basename(p)})
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        if not embeddings_list:
            print("Failed to generate any embeddings.")
            return

        all_embeddings = np.vstack(embeddings_list).astype('float32')
        
        dimension = all_embeddings.shape[1]
        print(f"Building FAISS index with dimension {dimension}...")
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(all_embeddings)
        
        self.save_index()
        print("Indexing complete.")

    def save_index(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        print(f"Index saved to {self.index_path}")

if __name__ == "__main__":
    indexer = Indexer()
    indexer.process_images("test")
