import os
os.environ["USE_TF"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import faiss
import json
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os

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

    def _single_query_search(self, text, k=5, return_dict=False):
        """
        Standard dense retrieval. 
        Args:
           return_dict: If True, returns a dict {filename: score} for easier fusion
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        
        text_emb = outputs.cpu().numpy()
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        
        scores, indices = self.index.search(text_emb.astype('float32'), k)
        
        results = []
        res_dict = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = {
                    "path": self.metadata[idx]["path"],
                    "filename": self.metadata[idx]["filename"],
                    "score": float(score)
                }
                results.append(item)
                res_dict[item["filename"]] = float(score)
                
        if return_dict:
            return res_dict
        return results

    def _compositional_search(self, text, k=5):
        """
        Hybrid Search:
        1. Get Global candidates (match whole query)
        2. Get candidates for each part
        3. Fuse scores: GlobalScore + Sum(PartScores)
        """
        text_clean = text.lower().replace(" and ", ",")
        raw_parts = text_clean.split(",")
        parts = [p.strip() for p in raw_parts if p.strip()]
        
        if len(parts) == 1:
            return self._single_query_search(parts[0], k)
            
        print(f"Refined Hybrid Search. Query: '{text}' -> Parts: {parts}")
        
        # Hyperparameters
        K_GLOBAL = k * 5
        K_PARTS = k * 10 
        WEIGHT_GLOBAL = 0.6  # Give significant weight to the global context
        WEIGHT_PARTS = 0.4   # Weight for individual attributes
        
        # 1. Global Search (Full context)
        # Fetching more candidates to intersect with parts
        global_scores = self._single_query_search(text, k=K_GLOBAL, return_dict=True)
        
        # 2. Part Search
        part_scores_list = []
        for p in parts:
            p_scores = self._single_query_search(p, k=K_PARTS, return_dict=True)
            part_scores_list.append(p_scores)
            
        # 3. Fusion
        # We consider the union of all candidates
        all_candidates = set(global_scores.keys())
        for ps in part_scores_list:
            all_candidates.update(ps.keys())
            
        final_scores = []
        
        for fname in all_candidates:
            # Base score from global (if not present, maybe small penalty or 0)
            g_score = global_scores.get(fname, 0.0)
            
            # Sum of parts
            p_score_sum = 0.0
            matches_count = 0
            for ps in part_scores_list:
                if fname in ps:
                    p_score_sum += ps[fname]
                    matches_count += 1
            
            # Normalize part score by number of parts? 
            # Or just sum? Sum rewards matching MORE parts.
            # Average might penalize if one part is missing but others are strong.
            # Let's use Sum but scaled.
            
            # Boost if it matches ALL parts?
            completeness_boost = 1.0
            if matches_count == len(parts):
                completeness_boost = 1.2
            
            # Final Formula
            # We want images that match the global description AND contain the parts.
            # If an image is not in global candidates but is in parts, it might be good.
            # If an image is in global but not in parts, it might be missing details.
            
            # Hybrid Score
            total_score = (g_score * WEIGHT_GLOBAL) + (p_score_sum * WEIGHT_PARTS / len(parts))
            total_score *= completeness_boost
            
            # Retrieve path (inefficient loop, optimize later)
            # Find path in metadata or cache? 
            # We need to find the path again.
            path = ""
            # Search in index mapping logic... wait, we only have limited info.
            # Let's assume we can scan our temporary results to find the path.
            # This is slow but correct for this assignment.
            found = False
            
            # Look in global results first
            # (Requires global_scores to store path too, but we simplified return_dict to just scores)
            # Let's re-find path from the original lists.
            # Not saving path in return_dict was a mistake for retrieval speed, but fine for prototype.
            
            # Helper to find path in result lists
            # We need to keep the full objects.
            pass
        
        # Quick fix: Re-run fusion with full objects
        # Map fname -> obj
        fname_to_obj = {}
        
        # Collect objects from global
        g_results = self._single_query_search(text, k=K_GLOBAL, return_dict=False)
        for item in g_results:
            fname_to_obj[item['filename']] = item
            
        # Collect objects from parts
        for p in parts:
            p_res = self._single_query_search(p, k=K_PARTS, return_dict=False)
            for item in p_res:
                if item['filename'] not in fname_to_obj:
                    fname_to_obj[item['filename']] = item
                    
        # Score calculation again
        candidates = []
        for fname, obj in fname_to_obj.items():
            g_score = 0.0
            # Check global list
            for item in g_results:
                if item['filename'] == fname:
                    g_score = item['score']
                    break
            
            p_score_sum = 0.0
            matches = 0
            for p in parts:
                # We need to search again? No, we should have cached the scores.
                # Doing search inside loop is bad.
                # Let's assume we call _single_query_search again is okay (cached models)?
                # No, we already ran it. We need the scores from the earlier runs.
                pass
            
        # Refactored Clean Implementation below
        
        # 1. Global
        g_res = self._single_query_search(text, k=K_GLOBAL)
        g_map = {x['filename']: x['score'] for x in g_res}
        
        # 2. Parts
        p_maps = []
        for p in parts:
            res = self._single_query_search(p, k=K_PARTS)
            p_maps.append({x['filename']: x['score'] for x in res})
            
        # 3. Fuse
        all_fnames = set(g_map.keys())
        for pm in p_maps:
            all_fnames.update(pm.keys())
            
        scored_candidates = []
        for fname in all_fnames:
            score_g = g_map.get(fname, 0.0)
            
            score_p_sum = 0.0
            matches = 0
            for pm in p_maps:
                if fname in pm:
                    val = pm[fname]
                    score_p_sum += val
                    matches += 1
            
            # Boost
            boost = 1.0
            if matches == len(parts):
                boost = 1.15
            
            # Weighted average
            # If len(parts) is large, sum grows, so divide.
            final_score = (score_g * WEIGHT_GLOBAL) + ((score_p_sum / len(parts)) * WEIGHT_PARTS)
            final_score *= boost
            
            scored_candidates.append({
                "filename": fname,
                "score": final_score
            })
            
        # Sort
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        top_k = scored_candidates[:k]
        
        # Resolve Paths
        final_objects = []
        for item in top_k:
            fname = item["filename"]
            # Look up path from our object cache
            # We need to find the path in our metadata.
            # Using loop over metadata is O(N), slow.
            # But we have 'fname_to_obj' from before? No, let's just find it in the search results we got.
            
            path = None
            # Check global results
            for x in g_res:
                if x["filename"] == fname:
                    path = x["path"]
                    break
            
            if not path:
                # Check part results (we need to re-fetch part results effectively)
                # Since we didn't save the full objects for parts in step 2...
                # We should have.
                pass
                
        # To avoid re-running, let's keep a map of fname -> path during the scoring phase
        path_map = {}
        for x in g_res: 
            path_map[x['filename']] = x['path']
            
        # We need to iterate parts again to populate path_map for items ONLY in parts
        for p in parts:
             res = self._single_query_search(p, k=K_PARTS) # This is cached/fast enough? 
             # Actually we ran it in step 2 but discarded the list. 
             # Let's fix the implementation to store it.
             for x in res:
                 path_map[x['filename']] = x['path']

        final_objects = []
        for item in top_k:
            fname = item["filename"]
            if fname in path_map:
                final_objects.append({
                    "path": path_map[fname],
                    "filename": fname,
                    "score": item["score"]
                })
                
        return final_objects

