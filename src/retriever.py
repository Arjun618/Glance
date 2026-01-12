import os
os.environ["USE_TF"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import faiss
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class Retriever:
    def __init__(self, index_path="data/index.faiss", metadata_path="data/metadata.json", model_name="patrickjohncyh/fashion-clip"):
        print("Loading resources for Retriever...")
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
        except:
            print("Fallback to vanilla CLIP in Retriever")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Loaded index with {self.index.ntotal} vectors.")

    def search(self, query_text, k=5, compositional=True):
        if compositional and (" and " in query_text.lower() or ", " in query_text.lower()):
            return self._compositional_search(query_text, k)
        return self._single_query_search(query_text, k)

    def _single_query_search(self, text, k=5):
        """
        Standard simple search.
        """
        # Encode
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        text_emb = outputs.cpu().numpy()
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        
        scores, indices = self.index.search(text_emb.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "path": self.metadata[idx]["path"],
                    "filename": self.metadata[idx]["filename"],
                    "score": float(score)
                })
        return results

    def _compositional_search(self, text, k=5):
        """
        Optimized Hybrid Search:
        1. Batch encode [Global Query, Part 1, Part 2, ...]
        2. Batch Search in FAISS
        3. Efficient Fusion
        """
        # 1. Parse parts
        text_clean = text.lower().replace(" and ", ",")
        raw_parts = text_clean.split(",")
        parts = [p.strip() for p in raw_parts if p.strip()]
        
        if len(parts) <= 1:
            return self._single_query_search(text, k)

        print(f"Batch Optim. Search: '{text}' -> Parts: {parts}")

        # 2. Batch Encode
        # [Global, Part1, Part2, ...]
        all_texts = [text] + parts
        
        inputs = self.processor(text=all_texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        # Normalize
        embeddings = outputs.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype('float32')

        # 3. Batch Search
        # Global gets less k? No, let's get ample global candidates
        K_GLOBAL = k * 5
        K_PARTS = k * 10
        
        # We need variable K for global vs parts? FAISS 'search' is fixed k.
        # Let's just search K_PARTS (larger) for everyone and slice later if needed.
        D, I = self.index.search(embeddings, K_PARTS)
        
        # D[0], I[0] -> Global results
        # D[1], I[1] -> Part 1 results
        # ...

        # 4. Efficient Fusion
        # Weights
        WEIGHT_GLOBAL = 0.6
        WEIGHT_PARTS = 0.4
        
        # Map: filename -> {global_score: float, part_matches: int, part_score_sum: float, path: str}
        candidates = {}
        
        # Helper to process results
        def add_candidate(idx_list, score_list, is_global=False):
            for i, score in zip(idx_list, score_list):
                if i == -1: continue
                
                fname = self.metadata[i]["filename"]
                fpath = self.metadata[i]["path"]
                
                if fname not in candidates:
                    candidates[fname] = {
                        "global_score": 0.0,
                        "part_matches": 0,
                        "part_score_sum": 0.0,
                        "path": fpath
                    }
                
                if is_global:
                    candidates[fname]["global_score"] = float(score)
                else:
                    candidates[fname]["part_matches"] += 1
                    candidates[fname]["part_score_sum"] += float(score)

        # Process Global (Row 0)
        # Slice to K_GLOBAL if we want strictly fewer, but more data is fine.
        add_candidate(I[0], D[0], is_global=True)
        
        # Process Parts (Rows 1..N)
        for row_idx in range(1, len(all_texts)):
            add_candidate(I[row_idx], D[row_idx], is_global=False)
            
        # 5. Score & Sort
        final_list = []
        num_parts = len(parts)
        
        for fname, data in candidates.items():
            g = data["global_score"]
            p_avg = data["part_score_sum"] / num_parts if num_parts > 0 else 0
            
            # Boost
            boost = 1.0
            if data["part_matches"] == num_parts:
                boost = 1.15
            
            final_score = (g * WEIGHT_GLOBAL) + (p_avg * WEIGHT_PARTS)
            final_score *= boost
            
            final_list.append({
                "path": data["path"],
                "filename": fname,
                "score": final_score
            })
            
        # Sort desc
        final_list.sort(key=lambda x: x["score"], reverse=True)
        
        return final_list[:k]
