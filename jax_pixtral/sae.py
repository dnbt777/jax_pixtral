#|==============================================================>
#|_______ sae.py ____________________________________
#|
#|
#|  Training and inference functions for Sparse Auto Encoders
#| 
#|  Functions
#|    - SAE functions: init, save, load
#|    - SAE training functions
#|
#|
#|===============================================================>>>

import jax
import jax.numpy as jnp
import jax.random as jrand
from einops import rearrange
from functools import partial

from PIL import Image

from jax_pixtral.forward_common import (
    vision_encoder, vision_language_adapter, text_embedding, multimodal_embedding,
    transformer_block
)
from jax_pixtral.forward_common import *
from jax_pixtral.forward_training import MSE

from typing import NamedTuple, List
from jax_pixtral.model_types import *
from jax_pixtral.lora_types import *

from safetensors.flax import save_file, load_file




class SparseAutoEncoder(NamedTuple):
    in_layer: jax.Array # one for each layer?? or..
    hidden_layer: jax.Array # one for each layer?? or..
    bias_pre: jax.Array
    bias_enc: jax.Array
    bias_dec: jax.Array


def init_sae(key, in_size, hidden_size, out_size):
    dtype = jnp.bfloat16
    in_key, hidden_key = jrand.split(key)
    return SparseAutoEncoder(
        in_layer=jrand.normal(in_key, (in_size, hidden_size), dtype=dtype)*dtype(jnp.sqrt(2.0/in_size)),
        hidden_layer=jrand.normal(hidden_key, (hidden_size, out_size), dtype=dtype)*dtype(jnp.sqrt(2.0/hidden_size)),
        bias_pre=jnp.zeros((in_size,), dtype=dtype),
        bias_enc=jnp.zeros((hidden_size,), dtype=dtype),
        bias_dec=jnp.zeros((out_size,), dtype=dtype),
    )



# to train the SAE:
# rather than inserting it mid-xfmr
# have the xfmr return activations at layer n
# then, once returned, train the SAE on them.

# add intervention code later



### Activations functions

# capture the residual stream
def partial_transformer_block(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d: jax.Array,
    query_heads: int, kv_heads: int, head_dim: int,
    attn_mask: jax.Array,
    residual: int = 1, # 0, 1, 2 - post-attn by default
) -> jax.Array:
    # same block for vision encoder AND transformer
    # attention norm
    residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
    if residual == 0:
        return hidden_state_BTC + residual_BTC
    # attention
    residual_BTC = pixtral_attention(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask)
    if residual == 1:
        return hidden_state_BTC + residual_BTC # this one?
    hidden_state_BTC = hidden_state_BTC + residual_BTC
    # ff norm
    residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
    residual_BTC = feed_forward(block_params, residual_BTC)
    if residual == 2:
        return hidden_state_BTC + residual_BTC # or this one?


def text_forward_activations(model_params: PixtralModel, batch_tokens, batch_attn_mask, sae_layer):
  hidden_state_BTC = text_embedding(model_params, batch_tokens)
  return forward_activations(model_params, hidden_state_BTC, batch_attn_mask, sae_layer)


def mm_forward_activations(model_params: PixtralModel, batch_tokens, batch_image_sets, batch_intext_image_start_indices, batch_attn_mask, sae_layer):
  hidden_state_BTC = multimodal_embedding(model_params, batch_tokens, batch_image_sets, batch_intext_image_start_indices)
  return forward_activations(model_params, hidden_state_BTC, batch_attn_mask, sae_layer)



# gets the forward activations at the layer 'sae_layer'
# these will be used for training the SAE.
# they will also be used on pretrained SAE to find the
@partial(jax.jit, static_argnames=["sae_layer"])
def forward_activations(model_params, hidden_state_BTC, batch_attn_mask, sae_layer):
  B, T, C = hidden_state_BTC.shape
  head_dim = 128 # params.json
  max_pos, d = T, head_dim
  freqs = precompute_rope_freqs_1d(max_pos, d) # mistral does rope after splitting k and q into gqa heads. q and k are split into the same channel size per head

  # attention layers
  Hq = 32 # params.json
  Hk = 8 # params.json
  attn_mask = get_causal_mask(T)[None, None, :, :]
  attn_mask = jnp.logical_or(batch_attn_mask[:, None, None, None, :], attn_mask)  # if True in either mask, mask out token
  # head dim defined above - it's used to calculate rope1d frequencies
  # scan compiles faster than a for loop
  def scanf(hidden_state, xfmr_block_num):
    xfmr_block_params = jax.tree_util.tree_map(lambda x: x[xfmr_block_num, ...], model_params.transformer.transformer_layers)
    hidden_state = transformer_block(xfmr_block_params, hidden_state, freqs, Hq, Hk, head_dim, attn_mask)
    return hidden_state, None
  hidden_state_BTC, _ = jax.lax.scan(
    scanf,
    hidden_state_BTC,
    jnp.arange(sae_layer-1), # scan up to this layer, then return the hidden state activations
  )

  # finally, complete the last layer to get the residual stream activations
  residual_stream_activations_BTC = partial_transformer_block(
      jax.tree_util.tree_map(lambda x: x[sae_layer, ...], model_params.transformer.transformer_layers),
      hidden_state_BTC,
      freqs, Hq, Hk, head_dim, attn_mask,
      residual=1 # capture post-attention residual
  )
    
  return residual_stream_activations_BTC




@partial(jax.jit, static_argnames=["sae_layer"])
def get_activations(
     pixtral_params: PixtralModel,
     batch_message_tokens: jax.Array,
     batch_attn_mask: jax.Array,
     sae_layer: int,
) -> float:
    residual_stream_activations_BTC = text_forward_activations(pixtral_params, batch_message_tokens, batch_attn_mask, sae_layer)
    return residual_stream_activations_BTC



@jax.jit
def text_lora_loss_fn(
     pixtral_params: PixtralModel,
     lora_params: LoRA,
     batch_message_tokens: jax.Array,
     batch_context_mask: jax.Array,
     batch_padding_mask: jax.Array,
) -> float:
    #pixtral_params = merge_lora(pixtral_params, lora_params) # not mem efficient in backwards
    # forward
    batch_input_tokens = batch_message_tokens[:, :-1]
    batch_target_tokens = batch_message_tokens[:, 1:]
    batch_attn_mask = batch_padding_mask[:, :-1] # align with inputs
    batch_next_token_logits = text_forward_train(pixtral_params, batch_input_tokens, batch_attn_mask, lora_params=lora_params)
    # mask out context tokens (i.e. only train on assistant's response)
    # mask out padding tokens (padding_mask) and user prompt tokens (context_mask)
    batch_loss_mask = jnp.logical_or(batch_context_mask, batch_padding_mask) # mask out context and padding, only grade on response
    batch_loss_mask = batch_loss_mask[:, 1:, None] # align with targets
    # get loss
    batch_loss = cross_entropy_loss(batch_next_token_logits, batch_target_tokens, batch_loss_mask)
    return batch_loss



#
# def intervention:
# transformer_block(..., sae_vector=(precomputed vector here. add this to residual stream if layer==sae_layer), sae_layer=n)


# do the full forward - encode and decode and get loss
def old_sae_forward(sae_params, residual_activations, l1=1e-3):
    # norm by mean over channel https://cdn.openai.com/papers/sparse-autoencoders.pdf
    mean = jnp.mean(residual_activations, axis=-1, keepdims=True)
    normalized = residual_activations - mean
    encoded = jax.nn.relu(normalized @ sae_params.in_layer)
    decoded = encoded @ sae_params.hidden_layer
    reconstruction = MSE(decoded, normalized)
    sparsity = l1*jnp.mean(jnp.abs(encoded))
    return reconstruction + sparsity


# do the full forward - encode and decode and get loss
def sae_forward(sae_params, residual_activations, l1=1e-3):
    # norm by mean over channel https://cdn.openai.com/papers/sparse-autoencoders.pdf
    mean = jnp.mean(residual_activations, axis=-1, keepdims=True)
    normalized = residual_activations - mean
    encoded = jax.nn.relu((normalized - sae_params.bias_pre) @ sae_params.in_layer + sae_params.bias_enc)
    decoded = encoded @ sae_params.hidden_layer + sae_params.bias_dec
    reconstruction = MSE(decoded, normalized)
    sparsity = l1*jnp.mean(jnp.abs(encoded))
    return reconstruction + sparsity



# get the activations in the SAE
# used for finding features
def sae_encode(sae_params, residual_activations):
    mean = jnp.mean(residual_activations, axis=-1, keepdims=True) # i think this has to be learned and feature-wise
    normalized = residual_activations - mean
    encoded = jax.nn.relu((normalized - sae_params.bias_pre) @ sae_params.in_layer + sae_params.bias_enc)
    return encoded # from here youd find the top_K

def sae_decode(sae_params, encoded_activations):
    return encoded_activations @ sae_params.hidden_layer + sae_params.bias_dec


class SAE(NamedTuple):
    sae_params: SparseAutoEncoder
    concept_vector: jax.Array
    layer: int




def sae_intervention(sae, residual_activations):
    activation_mean = jnp.mean(residual_activations, axis=-1, keepdims=True)
    encoded_activations = sae_encode(sae.sae_params, residual_activations)
    modified_activations = encoded_activations + sae.concept_vector # clamp in future. for now just add them ig
    decoded_activations = sae_decode(sae.sae_params, modified_activations) + activation_mean
    return decoded_activations







