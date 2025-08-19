###
### run_lora_inference.py
### --- loads a lora from ./loras, and runs inference.
### --- (you need to run run_lora_training.py first, to generate a lora.)
###

from jax_pixtral.inference import get_completions



# create prompt (context)
context = [
    {
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "Say hi!"
            },
        ],
    },
]



# run inference on prompt
completion = get_completion(
    context,
    max_tokens=64,
    temp=0.0,
    lora_path="loras/test.safetensors"
)

print(completion)
