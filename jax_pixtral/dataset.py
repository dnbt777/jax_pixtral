#|==============================================================>
#|_______ dataset.py _____________________________________
#|
#|
#|  Functions for loading/saving completion (JSON) datasets
#|
#|
#|
#|===============================================================>>>



import json



def load_dataset(path):
    with open(path) as f:
        return json.load(f)



def save_dataset(json_completions, path):
    with open(path, 'w') as f:
        json.dump(json_completions, f)
