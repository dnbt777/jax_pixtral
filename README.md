# Pixtral from scratch in jax

## features
- batch LoRA training and inference
- live chat (supports images, commands, LoRAs from training)
- simple code, simple api, and example scripts (run_*.py)


<img height="777" alt="image" src="https://github.com/user-attachments/assets/a9734d91-9028-42f5-ae75-8cad42d46f23" />


## what is this?

This is a VLM inference/training library

I made this for fun + learning + to do VLM/LLM experiments in jax

It is mostly from scratch, but not 100%. Besides jax, it uses:
- pillow: image processing
- regex: text processing (for the tokenizer)
- einops: makes the code easier to read and reduces bugs
- safetensors: load/save params

It's not perfect by any means. Improvement suggestions/feedback are very welcome

Optimization is WIP. Current throughput (A40): 
- single completion: 7 tok/s
- batched: 324 tok/s





## install/setup

Clone the repo and run setup.sh 

```
git clone https://github.com/dnbt777/jax_pixtral
cd jax_pixtral
./setup.sh [huggingface_key_for_downloading_pixtral_weights]
```


### Manual setup (if you dont want to run ./setup.sh [hf_key]):

1. install uv and sync packages
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2. Download pixtral weights+config from huggingface (~25GB)
```
HF_HOME="./pixtral"
uv run hf download mistralai/Pixtral-12B-2409 --local-dir ./pixtral --token "[hf token with read access]"
```




## run
from here, you can run/modify the following example scripts:
- run_chat.py
- run_inference.py
- run_batch_inference.py
- run_lora_training.py
- run_lora_inference.py



## environment
This was tested in the following environment:
- runpod A40, EU-SE-1 region, with the pytorch cuda12.4 template and 60GB storage volume

This costs ~$0.40/hr if you would like to mimic this exact setup.

Why use a 60GB volume?
- The default size doesn't have enough storage to download the pixtral weights
- Both "git lfs clone" and "huggingface-cli download" duplicate the weights while downloading.



## known bugs/limitations
- when streaming completions (i.e. in the chat), emojis dont render properly (see the header img in this readme)
- single-batch tok/sec is low (7tok/sec on an A40)
- OOM for ~4+ images in the chat (A40)
- support for models besides pixtral 12B not yet added
- it feels like attention leaks between images. maybe a masking issue


## features todo
- Multimodal LoRA training is WIP
- add optimizers (adam, muon)
- add more lora types (such as mlp, embeddings, etc)
- add batch data loading from json files


## optimizations todo
- currently re-does the entire prefill on each new chat message
- general profiling


## codebase style choices
- values/functions comment at the top of files:
    - it seems useful to be able to read a map of the code before reading the code.
- hardcoding params instead of loading from params.json
    - Does not matter
    - I will not fix this until I decide to support other pixtral models or support training from scratch








