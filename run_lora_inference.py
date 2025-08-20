###
### run_lora_inference.py
### --- loads a lora from ./loras, and runs inference.
### --- (you need to run run_lora_training.py first, to generate a lora.)
###

from jax_pixtral.inference import get_completions



# create prompt (context)
context1 = [
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

# create different prompt
context2 = [
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


prompts = [context1, context2]


# run inference on prompt
completions = get_completions(
    prompts,
    max_tokens=64,
    temp=0.0,
    lora_path="loras/test.safetensors",
    tokenizer_config_dir="./pixtral"
)

print(completions)