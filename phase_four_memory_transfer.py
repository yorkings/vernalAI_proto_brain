import torch
class TemplateMemory:
    def __init__(self, capacity=500, similarity_threshold=0.8):
        self.templates = {}  # fingerprint -> template data
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
    def compute_similarity(self, fp1, fp2):
        # Simple cosine similarity for fingerprint vectors
        return torch.cosine_similarity(fp1, fp2, dim=0).item()
    
    def find_similar(self, query_fp):
        best_similarity = 0
        best_template = None
        
        for stored_fp, template in self.templates.items():
            similarity = self.compute_similarity(query_fp, stored_fp)
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_template = template
        
        return best_template if best_template else None
    
    def store_template(self, fingerprint, chain, success_count=1):
        if len(self.templates) >= self.capacity:
            # Evict least successful
            worst_fp = min(self.templates.keys(), 
                          key=lambda k: self.templates[k]['success_count'])
            del self.templates[worst_fp]
        
        self.templates[fingerprint] = {
            'chain': chain,
            'success_count': success_count,
            'failure_count': 0
        }