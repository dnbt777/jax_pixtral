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

from jax_pixtral.forward_training import batch_parse_completions, init_lora, save_lora, text_lora_loss_fn, adam, muon
from jax_pixtral.inference import preloaded_get_completions

from jax_pixtral.dataset import load_dataset, save_dataset

import time # logging



seed = (0_0)-7
rolling_key = jrand.PRNGKey(seed)



### load data
completions = load_dataset("./datasets/match_case.dataset")

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
attn_rank = 4
layers = 40
k_proj_shape = (5120, 1024)
o_proj_shape = (4096, 5120)
q_proj_shape = (5120, 4096)
v_proj_shape = (5120, 1024)
# xfmr block lora
block_rank = 4
attn_norm_size = 5120
ffw_norm_size = 5120
ffw1_shape = (14336, 5120)[::-1]
ffw2_shape = (5120, 14336)[::-1]
ffw3_shape = (14336, 5120)[::-1] # transposed

# create the lora
lora_params = init_lora(
    rolling_key,
    # dense
    dense_in_dim, dense_out_dim, dense_rank,
    # attn lora
    q_proj_shape[0], q_proj_shape[1], attn_rank,
    k_proj_shape[0], k_proj_shape[1], attn_rank,
    v_proj_shape[0], v_proj_shape[1], attn_rank,
    o_proj_shape[0], o_proj_shape[1], attn_rank,
    # block lora
    attn_norm_size, ffw_norm_size,
    ffw1_shape[0], ffw1_shape[1], block_rank,
    ffw2_shape[0], ffw2_shape[1], block_rank,
    ffw3_shape[0], ffw3_shape[1], block_rank,
    # layer count
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
# init data
datapoints = batch_completions["tokens"].shape[0]
dataset_idxs = jnp.arange(datapoints)
split = 0.8
split = int(split*datapoints)
train_idxs, test_idxs = dataset_idxs[:split], dataset_idxs[split:]
batch_size = 32
train_batches = train_idxs.size // batch_size
epochs = 1
i = 0
# init adam
v = jax.tree_util.tree_map(lambda x: 0, lora_params)
s = jax.tree_util.tree_map(lambda x: 0, lora_params)
beta1 = 0.98 # direction momentum
beta2 = 0.99 # magnitude momentum
for epoch in range(epochs):
    for batch in range(train_batches):
        #batch = 0 # overfit test
        i += 1
        ## get train loss and grads
        start_idx, end_idx = batch*batch_size, (batch+1)*batch_size
        loss, grads = jax.value_and_grad(text_lora_loss_fn, argnums=1)(
            pixtral_params,
            lora_params, # arg 1
            batch_completions["tokens"][start_idx:end_idx],
            batch_completions["context_mask"][start_idx:end_idx],
            batch_completions["padding_mask"][start_idx:end_idx],
        )
        
        ## update (simple SGD)
        lr = 1e-3 #* (0.5 + jnp.abs(jnp.cos(i/20))) * (0.998**i)
        print(f"it: {i} || loss: {loss.item():.5f} || lr: {lr:.7f}")
        updates, v, s = adam(grads, v, s, i, lr, beta1, beta2)
        lora_params = jax.tree_util.tree_map(lambda p, dp: p + dp, lora_params, updates) # TODO implement adam, adamw, muon

        ## print grads (shows how params are learning) (uncomment to enable)
        #print(jax.tree_util.tree_map(lambda g: jax.numpy.linalg.norm(g), updates))

        # val loss
        if batch % 16 == 0: # just dont print this
            val_start_idx, val_end_idx = test_idxs[0], min(test_idxs[-1], test_idxs[0] + batch_size)
            val_loss = text_lora_loss_fn(
                pixtral_params,
                lora_params,
                batch_completions["tokens"][val_start_idx:val_end_idx],
                batch_completions["context_mask"][val_start_idx:val_end_idx],
                batch_completions["padding_mask"][val_start_idx:val_end_idx],
            )
            print(f"it: {i} || val_loss: {val_loss.item():.5f} || lr: {lr:.7f}")



### save lora for future use in inference/chat or in continued fine tuning
filepath = "loras/match_case.safetensors"
save_lora(lora_params, filepath)
print(f"Saved lora to {filepath}")



### test inference with lora
#completions = preloaded_get_completions(pixtral_params, [context], max_tokens=64, temp=0.0, lora_params=lora_params, tokenizer_config_dir="./pixtral")
#print(completions)

