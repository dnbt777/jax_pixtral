from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jrand

from jax_pixtral.load_model import load_params, fast_load_params

from jax_pixtral.forward_training import batch_parse_completions, init_lora, save_lora, text_lora_loss_fn, adam
from jax_pixtral.inference import preloaded_get_completions

import time # logging


from jax_pixtral.sae import *

# sources
# [0] https://cdn.openai.com/papers/sparse-autoencoders.pdf







# load dataset
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
                "text": "Say hi! Introduce yourself as a friendly assistant."
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
                "text": "Hello! I'm just a friendly assistant. How can I help you?"
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
print(batch_completions["tokens"].shape)



# load pixtral params
load_start = time.time()
print("loading params")
paths = ['./pixtral/consolidated.safetensors']
pixtral_params = fast_load_params(paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")


# get activations from residual stream [0]
# context length = 64 [0]
target_layer = 32 # out of 40. target later layers approximately 80% of the way through the model [0]
residual_activations = get_activations(pixtral_params, batch_completions["tokens"], batch_completions["padding_mask"], target_layer)


# init sae
key = jrand.PRNGKey(0)
channel_dim = residual_activations.shape[-1]
in_size, out_size = channel_dim, channel_dim
hidden_size = in_size*4

sae_params = init_sae(key, in_size, hidden_size, out_size)



## train sae on dataset's activations
def MSE(yhat, y):
    return jnp.mean((y - yhat)**2)


def sae_forward(sae_params, residual_activations, l1=1e-3):
    # divide by mean over channel https://cdn.openai.com/papers/sparse-autoencoders.pdf
    mean = jnp.mean(residual_activations, axis=-1, keepdims=True)
    normalized = residual_activations - mean
    encoded = jax.nn.relu(normalized @ sae_params.in_layer)
    decoded = encoded @ sae_params.hidden_layer
    reconstruction = MSE(decoded, normalized)
    sparsity = l1*jnp.mean(jnp.abs(encoded))
    return  reconstruction + sparsity
    



# train loop
v = jax.tree_util.tree_map(lambda x: 0, sae_params)
s = jax.tree_util.tree_map(lambda x: 0, sae_params)
beta1 = 0.98 # direction
beta2 = 0.99 # magnitude
lr = 1e-5
iterations = 100

for i in range(1, iterations+1):
    loss, grads = jax.value_and_grad(sae_forward)(sae_params, residual_activations)

    updates, v, s = adam(grads, v, s, i, lr, beta1, beta2)

    sae_params = jax.tree_util.tree_map(lambda p, u: p + u, sae_params, updates)

    print(loss)



# meh do this layer ig
# save trained sae (trained.sae)


























# load target concept dataset


# run dataset through sae-net and record activations


# find activated SAE neuron indexes and clamp to high number
top_k_neurons = 5 # idk lol