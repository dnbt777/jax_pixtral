###
### run_build_dataset.py
### --- builds a dataset for finetuning by generating completions and applying a function to alter them
### --- example: uppercase(get_completion(""))
###

from jax_pixtral.inference import get_completions
from jax_pixtral.dataset import load_dataset, save_dataset

import random as rand


# create prompts. make a dataset to fine-tune pixtral to respond in all caps if the user does.
words = "Hi Hello Hey Uh Sup So lmao what lololol hi".split()
prompts = []

# duplicate n times to create a batch of (BATCH_SIZE) prompts
BATCH_SIZE = 32
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
    



### run inference on the entire batch
completions = get_completions(prompts, max_tokens=64, temp=1.0,
                              tokenizer_config_dir="./pixtral", return_full_context=True)
print(completions[0])

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






