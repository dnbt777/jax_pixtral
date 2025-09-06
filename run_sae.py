from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jrand

from jax_pixtral.load_model import load_params, fast_load_params

from jax_pixtral.forward_training import batch_parse_completions, init_lora, save_lora, text_lora_loss_fn, adam
from jax_pixtral.inference import preloaded_get_completions

import time # logging


from jax_pixtral.sae import *
from jax_pixtral.forward_training import MSE


# sources
# [0] https://cdn.openai.com/papers/sparse-autoencoders.pdf



seed = 0



# load dataset
# generate batches of tokens of len=n from just <s>
from jax_pixtral.inference import *
BOS_TOKEN_ID = 1
max_tokens = 256
batches = 32
token_batches = [[BOS_TOKEN_ID] for _ in range(batches)]
batch_prompts = batch_format_token_prompts(token_batches, max_tokens)
#from jax_pixtral.forward import x



# load pixtral params
load_start = time.time()
print("loading params")
paths = ['./pixtral/consolidated.safetensors']
pixtral_params = fast_load_params(paths)
print(f"Loaded params in {time.time() - load_start:.2f}s")

# create vanilla completions to train SAE with
temp = 1.0
batch_completions = inference(pixtral_params, batch_prompts, max_tokens, temp, seed=seed)



# get activations from residual stream [0]
# context length = 64 [0]
target_layer = 32 # out of 40. target later layers approximately 80% of the way through the model [0]
residual_activations = get_activations(pixtral_params, batch_completions["tokens"], batch_completions["padding_mask"], target_layer)


# init sae
key = jrand.PRNGKey(seed+1)
channel_dim = residual_activations.shape[-1]
in_size, out_size = channel_dim, channel_dim
hidden_size = in_size*4

sae_params = init_sae(key, in_size, hidden_size, out_size)



# train loop
v = jax.tree_util.tree_map(lambda x: 0, sae_params)
s = jax.tree_util.tree_map(lambda x: 0, sae_params)
beta1 = 0.98 # direction
beta2 = 0.99 # magnitude
lr = 1e-3
iterations = 1000

for i in range(1, iterations+1):
    loss, grads = jax.value_and_grad(sae_forward)(sae_params, residual_activations)

    updates, v, s = adam(grads, v, s, i, lr, beta1, beta2)

    sae_params = jax.tree_util.tree_map(lambda p, u: p + u, sae_params, updates)

    print(i, loss)




# save trained sae (trained.sae)
    # meh do this later ig



# create target concept dataset
with open("./datasets/nix_sae.txt", 'r') as file: # random text i copied/pasted from nix website
    all_text = file.read()


# tokenize the text, break it into chunks of 64 tokens, and then get the activations
tokenizer, encode, decode = load_tokenizer(config_dir="./pixtral")
chunk_size = 64
all_tokens = encode(all_text)
token_chunks = [all_tokens[i:i+chunk_size-1] for i in range(0, len(all_tokens), chunk_size-1)] # (n, chunk_size-1)
token_chunks = [[BOS_TOKEN_ID] + _tokens for _tokens in token_chunks] # (n, chunk_size)
assert len(token_chunks[0]) == chunk_size

max_tokens = chunk_size
batch_chunks = batch_format_token_prompts(token_chunks, max_tokens)


chunk_count = batch_chunks["tokens"].shape[0]
batch_size = 32
batch_count = chunk_count // batch_size

for batch in range(batch_count):
    start_idx = batch*batch_size
    end_idx = min(start_idx + batch_size, chunk_count) # may cause func to be re-jitted on last batch. nbd
    residual_activations = get_activations(
        pixtral_params, 
        batch_chunks["tokens"][start_idx:end_idx], 
        batch_chunks["padding_mask"][start_idx:end_idx], 
        target_layer
    )
    print(f"got residual activations for batch {batch}")

    sae_activations_BTC = sae_encode(sae_params, residual_activations)
    if batch == 0:
        rolling_mean_sae_activation_TC = jnp.mean(sae_activations_BTC, axis=0)/batch_count
    else:
        rolling_mean_sae_activation_TC = rolling_mean_sae_activation_TC + jnp.mean(sae_activations_BTC, axis=0)/batch_count



# find activated SAE neuron indexes and clamp to high number
mean_sae_activation_C = jnp.mean(rolling_mean_sae_activation_TC, axis=0)
top_k_neurons = 5 # idk lol
# print the distribution skew (to measure sparsity. all neurons activated evenly == failure)
    # the top 20% of values account for .. how much of the activations?
    # note: all activations are positive (they are post-relu)
activation_count = mean_sae_activation_C.shape[-1]
split = 0.2
split_idx = int(activation_count*split)
sorted_sae_activations_C = jnp.sort(mean_sae_activation_C, descending=True)
top_sae_activation_mass = jnp.sum(sorted_sae_activations_C[:split_idx]) / jnp.sum(sorted_sae_activations_C)
print(f"the top 20% of activations are responsible for {top_sae_activation_mass.item()*100:2.2f}% of the total")


# decode using SAE params to get the vector to be added during inference
concept_vector = mean_sae_activation_C
concept_vector = jnp.where(
    concept_vector >= sorted_sae_activations_C[top_k_neurons],
    concept_vector,
    jnp.zeros_like(concept_vector),
)
print("filtering activation value:", sorted_sae_activations_C[top_k_neurons])


# assert - does doing nothing do nothing?
concept_vector = jnp.zeros_like(concept_vector) # DELETE AFTER TESTING


# run inference
prompt = "Name a random piece of software."
print(prompt)
prompt = [
    {"role" : "user", "content" : [{ "type" : "text", "text": prompt}]}
]


sae = SAE(
    sae_params=sae_params,
    concept_vector=concept_vector,
    layer=target_layer
)


max_tokens = 64
completion = preloaded_get_completion(
    pixtral_params,
    prompt,
    max_tokens,
    tokenizer_config_dir="./pixtral",
    sae=sae, # TODO IMPLEMENT
)
print("nixtral: ", completion)






