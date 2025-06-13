import os 
import pickle 

"""
for saving the results and loading them to save time 
in the calculation
"""

def save_cache(data,filepath):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    with open(filepath,'wb') as f:
        pickle.dump(data,f)
    

def load_cache(filepath):
    if os.path.exists(filepath):
        with open(filepath,'th') as f:
            return pickle.load()
    return None
