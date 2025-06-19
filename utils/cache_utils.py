import os 
import pickle 

"""
for saving the results for every feature of a dataset to save time in the calculations
"""

def save_cache(data,filepath):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    with open(filepath,'wb') as f:
        pickle.dump(data,f)
    

def load_cache(filepath):
    if os.path.exists(filepath):
        with open(filepath,'rb') as f:
            return pickle.load(f)
    return None

def compute_or_load_feature(feature_name,dataset_name,compute_fn,base_dir='cache',force_recompute=False):
    path = os.path.join(base_dir,dataset_name,f"{feature_name}.pkl")
    
    if not force_recompute:
        cached = load_cache(path)
        if cached is not None:
            print(f"[Cache] loaded: {feature_name}")
            return cached 
    
    print(f"[Compute] Computing : {feature_name}")
    result = compute_fn()
    save_cache(result,path)
    return result
