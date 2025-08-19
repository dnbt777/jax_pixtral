# Pixtral in jax from scratch (WIP)

## features
- live chat (supports images, commands)
- LoRA batch training and inferencing (including in chat)
- simple API for completions (get_completions(...))


<img height="777" alt="image" src="https://github.com/user-attachments/assets/a9734d91-9028-42f5-ae75-8cad42d46f23" />


## what is this?

This is a VLM inference/training library made for fun + learning + to do VLM/LLM experiments in jax

It is mostly from scratch, but not 100%. Besides jax, it uses:
- pillow: image processing
- regex: text processing (for the tokenizer)
- einops: makes the code easier to read and reduces bugs
- safetensors: load/save params

It's not perfect by any means, and improvement suggestions/feedback are very welcome






## install/setup

Clone the repo and run setup.sh 

```
git clone https://github.com/dnbt777/jax_pixtral
cd jax_pixtral
chmod +x setup.sh
./setup.sh
```

setup.sh:
- downloads and installs uv
- downloads pixtral weights from hf
- syncs uv (installs necessary packages)



## run
from here, you can run/modify the following example scripts:
- scripts/run_chat.py
- scripts/run_inference.py
- scripts/run_batch_inference.py
- scripts/run_lora_training.py
- scripts/run_lora_inference.py



## environment
This was tested on an A40 on runpod in the EU-SE-1 region with the pytorch cuda12.4 pod.

This costs $0.40/hr if you would like to mimic this exact setup.



## known bugs
- when streaming completions (i.e. in the chat), emojis dont render properly (see the header img in this readme)
- single-batch tok/sec is low (7tok/sec on an A40)
- OOM for ~4+ images in the chat


## features todo
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







