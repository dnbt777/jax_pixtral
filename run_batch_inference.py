###
### run_batch_inference.py
### --- runs inference on a batch of prompts of arbitrary sizes
###

from jax_pixtral.inference import get_completions



### create prompts of different sizes
prompts = []

# create first prompt (small)
context1 = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": "What is 1+1?",
          },
      ],
  },
]
prompts.append(context1)

# create second prompt (large)
prompt = """William Shakespeare[a] (c. 23 April 1564[b] – 23 April 1616)[c] was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the "Bard of Avon" or simply "the Bard". His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright. Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted."""
context2 = [
  {
      "role":
      "user",
      "content": [
          {
              "type": "text",
              "text": prompt
          },
      ],
  },
]
prompts.append(context2)

# duplicate n times to create a batch of (BATCH_SIZE + 2)
BATCH_SIZE = 50 
for i in range(BATCH_SIZE):
    prompts.append(context2)



### run inference on the entire batch
completions = get_completions(prompts, max_tokens=16, temp=0.0, tokenizer_config_dir="./pixtral")#, lora_path="loras/test.safetensors")



### print completions
for i, completion in enumerate(completions):
    print(f"{i}------------")
    print(completion)
    print(f"------------{i}")






