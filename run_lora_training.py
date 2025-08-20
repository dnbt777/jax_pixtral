###
### run_lora_training.py
### --- example of training a test lora on batches of messages
### --- fills the 'completions' list with a BATCH_SIZE messages, and overfits on it.
### --- saves the lora to ./loras, and runs inference on an example
###

import jax
import jax.numpy as jnp
import jax.random as jrand

from jax_pixtral.load_model import load_params, fast_load_params

from jax_pixtral.forward_training import batch_parse_completions, init_lora, save_lora, text_lora_loss_fn
from jax_pixtral.inference import preloaded_get_completions

import time # logging



seed = (0_0)-7
rolling_key = jrand.PRNGKey(seed)



### load data
### --- here we use a single message as a test
completions = []
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
response = [
    {
        "role":
        "assistant",
        "content": [
            {
                "type": "text",
                "text": "nah lmao i aint doin that" # fine tune it to say this!
            },
        ],
    },
]
completion = context + response

# create list of n completions that each end with a single assistant message
BATCH_SIZE = 1
for _ in range(BATCH_SIZE):
    completions.append(completion)

# stitch the completions together
# array-of-structs -> struct-of-(jax)-arrays (very jax friendly)
# in get_completions this happens automatically, but in training it does not
batch_completions = batch_parse_completions(completions, tokenizer_config_dir="./pixtral")



### initialize lora
## -- the LoRA namedtuple contains all lora types simultaneously
## -- set rank=0 for types you want to disable
# dense lora
dense_rank = 0 # set rank=0 to disable
channel_dim = 5120 # from pixtral
vocab_size = 131072 # from pixtral
dense_in_dim = channel_dim
dense_out_dim = vocab_size
# attn (qkvo) lora
attn_rank = 1
layers = 40
k_proj_shape = (5120, 1024)
o_proj_shape = (4096, 5120)
q_proj_shape = (5120, 4096)
v_proj_shape = (5120, 1024)
# create the lora
lora_params = init_lora(
    rolling_key,
    dense_in_dim, dense_out_dim, dense_rank,
    q_proj_shape[0], q_proj_shape[1], attn_rank,
    k_proj_shape[0], k_proj_shape[1], attn_rank,
    v_proj_shape[0], v_proj_shape[1], attn_rank,
    o_proj_shape[0], o_proj_shape[1], attn_rank,
    layers,
)
rolling_key, _ = jrand.split(rolling_key) # reroll key every time it's used



### load pixtral params
load_start = time.time()
print("loading params")
paths = ['./pixtral/consolidated.safetensors']
pixtral_params = fast_load_params(paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")



#### begin fine-tuning
# hyperparameters and other fine tuning advice:
# https://mistral.ai/news/unlocking-potential-vision-language-models-satellite-imagery-fine-tuning
for i in range(100):
    ## get loss and grads
    loss, grads = jax.value_and_grad(text_lora_loss_fn, argnums=1)(
        pixtral_params,
        lora_params, # arg 1
        batch_completions["tokens"],
        batch_completions["context_mask"], batch_completions["padding_mask"],
        rolling_key
    )
    rolling_key, _ = jrand.split(rolling_key)
    
    ## print grads (shows how params are learning) (uncomment to enable)
    # print(jax.tree_util.tree_map(lambda g: jax.numpy.linalg.norm(g), grads))
    
    ## update (simple SGD)
    lr = 1e-3 * (0.5 + jnp.abs(jnp.cos(i/20))) * (0.995**i)
    print(f"it: {i} || loss: {loss:.5f} || lr: {lr:.7f}")
    lora_params = jax.tree_util.tree_map(lambda p, g: p - lr*g, lora_params, grads) # TODO implement adam, adamw, muon



### save lora for future use in inference/chat or in continued fine tuning
filepath = "loras/test.safetensors"
save_lora(lora_params, filepath)
print(f"Saved lora to {filepath}")



### test inference with lora
completions = preloaded_get_completions(pixtral_params, [context], max_tokens=64, temp=0.0, lora_params=lora_params, tokenizer_config_dir="./pixtral")
print(completions)



