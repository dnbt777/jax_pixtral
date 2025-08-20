###
### run_chat.py
### --- example of how to run the chat
###

from jax_pixtral.inference import chat



chat(
    #lora_path="loras/test.safetensors",     # uncomment to enable lora, if youve trained one with run_lora_training.py
    verbose=True,                            # debug messages, such as "loading params" (recommended)
)
