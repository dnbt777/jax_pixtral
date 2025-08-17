# jax_pixtral
Pixtral in jax from scratch

# setup

Clone the repo and run setup.sh 

```
git clone https://github.com/dnbt777/jax_pixtral
cd jax_pixtral
chmod +x setup.sh
./setup.sh
```

setup.sh:
    downloads and installs uv
    downloads pixtral weights from hf
    syncs uv (installs necessary packages)



# run
from here, you can run/modify the following example scripts:
    scripts/run_chat.py
    scripts/run_inference.py
    scripts/run_batch_inference.py
    scripts/run_lora_training.py
    scripts/run_lora_inference.py


# environment

This was tested on an A40 on runpod in the EU-SE region with the pytorch cuda12.4 pod.

