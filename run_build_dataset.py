###
### run_build_dataset.py
### --- builds a dataset for finetuning by generating completions and applying a function to alter them
### --- example: uppercase(get_completion(""))
###

<<<<<<< HEAD
from jax_pixtral.inference import get_completions, preloaded_get_completions
from jax_pixtral.dataset import load_dataset, save_dataset
from jax_pixtral.load_model import load_params


import random as rand
import requests

import time


# create prompts. make a dataset to fine-tune pixtral to respond in all caps if the user does.
words = requests.get("https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt").text.splitlines()
words = words[200:] # skip garbage words (see file)
print("Downloaded dictionary")
print(",".join(words[:10]), "...")
=======
from jax_pixtral.inference import get_completions
from jax_pixtral.dataset import load_dataset, save_dataset

import random as rand


# create prompts. make a dataset to fine-tune pixtral to respond in all caps if the user does.
words = "Hi Hello Hey Uh Sup So lmao what lololol hi".split()
>>>>>>> bdeaa17960f401922f1fb29b4881b4d2f630bb2e
prompts = []

# duplicate n times to create a batch of (BATCH_SIZE) prompts
BATCH_SIZE = 32
<<<<<<< HEAD
BATCH_COUNT = 64
batches = []
for _ in range(BATCH_COUNT):
    batch = []
    for i in range(BATCH_SIZE):
        random_word = rand.choice(words)
        is_upper = rand.random() > 0.5
        if is_upper:
            random_word = random_word.upper()
        batch.append([
          {
              "role":
              "user",
              "content": [
                  {
                      "type": "text",
                      "text": random_word,
                  },
              ],
          },
        ])
    batches.append(batch)
print(f"Created {len(batches)} batches of {len(batches[0])} prompts")
=======
for i in range(BATCH_SIZE):
    random_word = rand.choice(words)
    is_upper = rand.random() > 0.5
    if is_upper:
        random_word = random_word.upper()
    prompts.append([
      {
          "role":
          "user",
          "content": [
              {
                  "type": "text",
                  "text": random_word,
              },
          ],
      },
    ])
    
>>>>>>> bdeaa17960f401922f1fb29b4881b4d2f630bb2e



### run inference on the entire batch
<<<<<<< HEAD
## preload params
load_start = time.time()
print("Loading params...")
safetensors_paths = ['./pixtral/consolidated.safetensors']
pixtral_params = load_params(safetensors_paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")


## generate completions to prompts
completions = []
for i, batch in enumerate(batches):
    batch_completions = preloaded_get_completions(pixtral_params, batches[i], max_tokens=32, temp=1.0,
                              tokenizer_config_dir="./pixtral", return_full_context=True)
    completions = completions + batch_completions
    print(f"added {BATCH_SIZE} completions. {i}/{BATCH_COUNT}")
print(len(completions))
=======
completions = get_completions(prompts, max_tokens=64, temp=1.0,
                              tokenizer_config_dir="./pixtral", return_full_context=True)
print(completions[0])
>>>>>>> bdeaa17960f401922f1fb29b4881b4d2f630bb2e

### transform completions
def assistant_response_transform(completion):
    text = completion[0]["content"][0]["text"]
    if text == text.upper():
        completion[-1]["content"][0]["text"] = completion[-1]["content"][0]["text"].upper()
    return completion



### dataset = map(f, completions)
dataset = [assistant_response_transform(completion) for completion in completions]

save_dataset(dataset, "./datasets/match_case.dataset")

dataset_ = load_dataset("./datasets/match_case.dataset")


### print example completion
for i, completion in enumerate(dataset_):
    print(f"{i}------------")
    print(completion)
    print(f"------------{i}")






