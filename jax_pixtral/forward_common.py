#|==============================================================>
#|_______ forward_common.py _____________________________________
#|
#|
#|  Common functions
#| 
#|  Functions
#|    - Pixtral forward helper functions
#|    - Embedding functions
#|    - RoPE functions
#|    - Mask creating functions
#|    - Common math functions (layernorm, rmsnorm, etc)
#|
#|  Conventions:
#|    - B: batch dim
#|    - T: time dim (the position of each token in the sequence)
#|    - C: channel dim (position inside each token's embedding)
#|    - d: max(channel_dim)
#|
#|
#|===============================================================>>>


import jax
import jax.numpy as jnp
import jax.random as jrand
from einops import rearrange
from functools import partial

from PIL import Image
from jax_pixtral.preprocessing import *

from jax_pixtral.lora_types import LoRA
from typing import List, Tuple
from jax_pixtral.model_types import *



### Pixtral forward helper functions

def pixtral_attention(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs: jax.Array,
    query_heads: int, kv_heads: int, head_dim: int,
    attn_mask: jax.Array,
    block_lora_params=None
) -> jax.Array:
    # compute qkv
    Q = hidden_state_BTC @ block_params.attention_wq_weight.T
    K = hidden_state_BTC @ block_params.attention_wk_weight.T
    V = hidden_state_BTC @ block_params.attention_wv_weight.T
    
    if block_lora_params:
        Q = Q + block_lora_params.attn.alpha_q*(
                (hidden_state_BTC @ block_lora_params.attn.in_q) @ block_lora_params.attn.out_q)
        K = K + block_lora_params.attn.alpha_k*(
                (hidden_state_BTC @ block_lora_params.attn.in_k) @ block_lora_params.attn.out_k)
        V = V + block_lora_params.attn.alpha_v*(
                (hidden_state_BTC @ block_lora_params.attn.in_v) @ block_lora_params.attn.out_v)
    
    # split into heads (GQA)
    Hk=kv_heads
    Hq=query_heads
    repeats = Hq // Hk 
    Q = rearrange(Q, "B T (Hk r d) -> B Hk r T d", Hk=Hk, r=repeats, d=head_dim)
    K = rearrange(K, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
    V = rearrange(V, "B T (Hk d) -> B Hk T d", Hk=Hk, d=head_dim)
    
    # rope1D AFTER splitting into GQA heads
    Q = apply_rope(Q, freqs)
    K = apply_rope(K, freqs) # both have the same head dim
    
    K = K[:, :, None, :, :] # broadcast over r
    V = V[:, :, None, :, :] # broadcast over r
    
    # mistral uses xformers memory efficient attention
    # https://github.com/facebookresearch/xformers/blob/e1a17a9235206dc7cd5999ce65ce79ff3cd4665d/xformers/ops/fmha/__init__.py#L194
    scale = jnp.bfloat16(1.0 / jnp.sqrt(Q.shape[-1]))
    Q = Q*scale
    attn = Q @ jnp.swapaxes(K, -1, -2)
    attn = jnp.where(attn_mask, jnp.bfloat16(-jnp.inf), attn) 
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(jnp.bfloat16) # do attn softmax in float32
    attn = attn @ V
    
    # collapse heads and outproject
    attn = rearrange(attn, "B H r T d -> B T (H r d)")
    out = attn @ block_params.attention_wo_weight.T
    if block_lora_params:
        out = out + block_lora_params.attn.alpha_o*(
                (attn @ block_lora_params.attn.in_o) @ block_lora_params.attn.out_o)
    
    return out.astype(jnp.bfloat16)



# https://github.com/mistralai/mistral-inference/blob/6eb35510403825cfb430b0004443053e8c4b70dc/src/mistral_inference/transformer_layers.py#L123
def transformer_block(
    block_params: TransformerBlock,
    hidden_state_BTC: jax.Array,
    freqs_1d: jax.Array,
    query_heads: int, kv_heads: int, head_dim: int,
    attn_mask: jax.Array,
    block_lora_params=None
) -> jax.Array:
    # same block for vision encoder AND transformer
    # attention norm
    if block_lora_params:
        residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight + block_lora_params.block.attnnorm)
    else:
        residual_BTC = RMSnorm(hidden_state_BTC, block_params.attention_norm_weight) ## attention norm
    # attention
    residual_BTC = pixtral_attention(block_params, residual_BTC, freqs_1d, query_heads, kv_heads, head_dim, attn_mask, block_lora_params=block_lora_params)
    hidden_state_BTC = hidden_state_BTC + residual_BTC
    # ff norm
    if block_lora_params:
        residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight + block_lora_params.block.ffwnorm)
    else:
        residual_BTC = RMSnorm(hidden_state_BTC, block_params.ffn_norm_weight) ## ff norm
    
    if block_lora_params:
        residual_BTC = feed_forward_lora(block_params, block_lora_params, residual_BTC)
    else:
        residual_BTC = feed_forward(block_params, residual_BTC)
    hidden_state_BTC = hidden_state_BTC + residual_BTC
    return hidden_state_BTC



# https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py
def vision_encoder(
    model_params: PixtralModel,
    processed_images: List[jax.Array],
    prefill: bool=False
) -> jax.Array:
    flattened_patch_embeddings_list, rope_2d_freqs = zip(*[embeddings_and_freqs(model_params, processed_image) for processed_image in processed_images])
    flattened_patch_embeddings_PC = jnp.concatenate(flattened_patch_embeddings_list, axis=0)
    rope_2d_freqs = jnp.concatenate(rope_2d_freqs, axis=1)
    
    # block diagonal mask
    attn_mask = create_block_diagonal_mask(flattened_patch_embeddings_list)
    
    # vision transformer blocks
    hidden_state_BTC = flattened_patch_embeddings_PC[jnp.newaxis, ...] # fake batch
    _, T, C = hidden_state_BTC.shape
    num_attention_heads = 16 # params.json
    hidden_dim = hidden_state_BTC.shape[-1]
    head_dim = hidden_dim // num_attention_heads
    
    def scanf(hidden_state, block_params):
        hidden_state = transformer_block(block_params, hidden_state, rope_2d_freqs, num_attention_heads, num_attention_heads, head_dim, attn_mask)
        return hidden_state, None
    hidden_state_BTC, _ = jax.lax.scan(scanf, hidden_state_BTC, model_params.vision_encoder.vision_encoder_layers)
    hidden_state_TC = hidden_state_BTC[0] # un-batch fake batch
    
    # vision language adapter
    hidden_state_TC = vision_language_adapter(model_params.vision_language_adapter, hidden_state_TC)
    return hidden_state_TC



def vision_language_adapter(vla_params: VisionLanguageAdapter, hidden_state_TC: jax.Array) -> jax.Array:
    hidden_state_TC = jax.nn.gelu(hidden_state_TC @ vla_params.w_in_weight.T + vla_params.w_in_bias)
    hidden_state_TC = hidden_state_TC @ vla_params.w_out_weight.T + vla_params.w_out_bias
    return hidden_state_TC



### Embedding functions

@jax.jit
def text_embedding(model_params: PixtralModel, text_tokens_batch: jax.Array) -> jax.Array:
    text_tokens_batch = jnp.array(text_tokens_batch, dtype=int) # B, T
    embeddings_BTC = jnp.take(model_params.tok_embeddings_weight, text_tokens_batch, axis=0) # C, B, T
    return embeddings_BTC



def embeddings_and_freqs(
    model_params: PixtralModel,
    processed_image: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    patch_embeddings_CHW = conv2d(model_params, processed_image)[0]
    C, H, W = patch_embeddings_CHW.shape
    num_attention_heads = 16 # params.json
    hidden_dim = C
    head_dim = hidden_dim // num_attention_heads
    freqs_2d = precompute_rope_freqs_2d(H, W, head_dim) # in the future - precompute with max H and W ONCE, and adjust func to deal with any H W array
    freqs_2d = rearrange(freqs_2d, "B H W c -> B (H W) c") # flatten
    # flatten patch embeddings
    flattened_patch_embeddings_PC = rearrange(patch_embeddings_CHW, "C H W -> (H W) C")
    # ln pre (RMSnorm)
    flattened_patch_embeddings_PC = RMSnorm(flattened_patch_embeddings_PC, model_params.vision_encoder.ln_pre_weight)
    return flattened_patch_embeddings_PC, freqs_2d



# probably the biggest bottleneck. needs heavy optimization
# this only runs once for prefill
# in the future, when we want to add prefill on top of an existing kvcache (i.e. user -> assistant -> 2nd user response w images),
# this will need to be optimized
def multimodal_embedding(
    model_params: PixtralModel,
    batch_message_tokens: jax.Array,
    batch_image_sets: List[List[jax.Array]],
    image_intext_start_indices_batches: jax.Array,
) -> jax.Array:
    # gets the embeddings of the tokens
    # already the exact length needed for images. contains img tokens including img_br and img_end
    text_embeddings_batch = text_embedding(model_params, batch_message_tokens) # BTC
    
    # get image embeddings
    image_embeddings_batch = [vision_encoder(model_params, image_set) for image_set in batch_image_sets]
    
    # replace img token placeholders with images
    patches_C = image_embeddings_batch[0].shape[-1]
    patch_size = 16 # params.json
    img_break_token_id, img_end_token_id = 1, 2
    img_break_embed = model_params.tok_embeddings_weight[img_break_token_id]
    img_end_embed = model_params.tok_embeddings_weight[img_end_token_id]
    for image_batch in range(len(image_intext_start_indices_batches)):
        inimg_start_idx = 0
        image_intext_start_indices = image_intext_start_indices_batches[image_batch]
        image_embeddings = image_embeddings_batch[image_batch]
        for i, intext_start_idx in enumerate(image_intext_start_indices):
            pixels_C, pixels_H, pixels_W = batch_image_sets[image_batch][i].shape
            patches_H, patches_W = pixels_H//patch_size, pixels_W//patch_size
            inimg_end_idx = inimg_start_idx + patches_H*patches_W # size(unformatted patches) = img patches H * img patches W
            intext_end_idx = intext_start_idx + patches_H*(patches_W + 1) # size(final patches) = img patches + break tokens + end token
            # there are two start indexes we will need. one is for the image inside of the image_embeddings. the other is where the image starts in the text_embeddings.
            image_embedding_TC = image_embeddings[inimg_start_idx:inimg_end_idx, ...]
            image_embedding_HWC = jnp.reshape(image_embedding_TC, (patches_H, patches_W, patches_C))
            # add the embeddings for img_break and img_end
            img_formatting_tokens = jnp.repeat(img_break_embed[None, None, :], patches_H, axis=0) # add img break tokens to each row
            img_formatting_tokens = img_formatting_tokens.at[-1, 0, :].set(img_end_embed) # replace last row's break token with an end token
            formatted_image_embedding_HWC = jnp.concatenate([image_embedding_HWC, img_formatting_tokens], axis=1)
            formatted_image_embedding_TC = jnp.reshape(formatted_image_embedding_HWC, (patches_H*(patches_W+1), patches_C))
            
            text_embeddings_batch = jax.lax.dynamic_update_slice(
                text_embeddings_batch, # dest
                formatted_image_embedding_TC[None, :], # source # bfloat16 just for now
                (image_batch, intext_start_idx, 0) # overwrite index = starting index in text tokens
            )
            inimg_start_idx = inimg_end_idx
    
    return text_embeddings_batch.astype(jnp.bfloat16)




### Rope functions

# todo: don't convert to complex, directly send sin and cos like in cached forward.
# otherwise its a waste of computation
@partial(jax.jit, static_argnames=["max_i", "d"])
def precompute_rope_freqs_1d(max_i: int, d: int) -> jax.Array:
    rope_theta = 1_000_000_000 # params.json
    max_j = d//2
    i, j = jnp.arange(max_i), jnp.arange(max_j)
    freqs_ij = jnp.outer(i, rope_theta**(-2.0*j/d)) # i, d//2
    cos, sin = jnp.cos(freqs_ij)[None, :], jnp.sin(freqs_ij)[None, :]
    return jax.lax.complex(cos, sin).astype(jnp.complex64)



@jax.jit
def apply_rope(hidden_state: jax.Array, freqs: jax.Array) -> jax.Array:
    original_shape = hidden_state.shape # ..., T, C
    T, C = hidden_state.shape[-2:]
    d = C
    hidden_state_pairs = jnp.reshape(hidden_state, (*hidden_state.shape[:-2], T, d//2, 2))
    re, im = hidden_state_pairs[..., 0], hidden_state_pairs[..., 1] # b, i, d//2
    # rotate each pair by its corresponding theta:
    # rotating a vector <x,y> by theta == multiplying (x + iy) by e^(i*theta)
        # where i = sqrt(-1), and x = re(al), and y = im(aginary) components
    # derivation:
    # complex rotation factor = e^(i*theta) = cos(theta) + i*sin(theta)
    # so (x + i*y) * e^(i*theta) =
    # = (x + i*y) * (cos(theta) + i*sin(theta))
    # = x*cos(theta) + i*y*cos(theta) + x*i*sin(theta) + i*y*i*sin(theta)
    # = x*cos(theta) - y*sin(theta) + i*(y*cos(theta) + x*sin(theta))
    # = <x*cos(theta) - y*sin(theta), y*cos(theta) + x*sin(theta)>
    cos, sin = jnp.real(freqs), jnp.imag(freqs)
    re_rot = re*cos - im*sin
    im_rot = im*cos + re*sin
    #hidden_state_pairs_rot = jnp.concatenate([re_rot, im_rot], axis=-1) # (B, T, d//2, 2)
    hidden_state_rot = jnp.zeros_like(hidden_state)
    hidden_state_rot = hidden_state_rot.at[..., ::2].set(re_rot.astype(jnp.bfloat16))
    hidden_state_rot = hidden_state_rot.at[..., 1::2].set(im_rot.astype(jnp.bfloat16)) # [re, im, re, im, re, im, ... ]
    return hidden_state_rot.astype(jnp.bfloat16)



@partial(jax.jit, static_argnames=["max_h", "max_w", "d"])
def precompute_rope_freqs_2d(max_h: int, max_w: int, d: int) -> jax.Array:
    rope_theta = 10_000 # params.json
    max_j = d//2
    h, w, j = jnp.arange(max_h), jnp.arange(max_w), jnp.arange(0, d, 2, dtype=jnp.bfloat16) # mimics -2.0 * (j from 0 to d//2)
    aten_div = j/d
    aten_pow = rope_theta**aten_div
    aten_pow = aten_pow
    base_freqs = jnp.reciprocal(aten_pow)
    thetas_hj = jnp.outer(h, base_freqs[::2]) # H, d//2
    thetas_hj = thetas_hj[:, jnp.newaxis, :] # H, d//2 => H, 1, d//2
    thetas_wj = jnp.outer(w, base_freqs[1::2]) # W, d//2
    thetas_wj = thetas_wj[jnp.newaxis, :, :] # 1, W, d//2
    
    # calculate hpos based rotations
    # pixtral uses aten::polar, which is just 1.0*cos(theta), 1.0*sin(theta)
    cos_h, sin_h = jnp.cos(thetas_hj)[jnp.newaxis, :].astype(jnp.float32), jnp.sin(thetas_hj)[jnp.newaxis, :].astype(jnp.float32) # add batch dim to both
    freqs_h = jax.lax.complex(cos_h, sin_h).astype(jnp.complex64) # 1, H, 1, d//4 # take the evens
    cos_w, sin_w = jnp.cos(thetas_wj)[jnp.newaxis, :].astype(jnp.float32), jnp.sin(thetas_wj)[jnp.newaxis, :].astype(jnp.float32)
    freqs_w = jax.lax.complex(cos_w, sin_w).astype(jnp.complex64) # 1, 1, W, d//4 # take the odds

    # make frequency grid
    freqs_w = jnp.repeat(freqs_w, repeats=max_h, axis=1)
    freqs_h = jnp.repeat(freqs_h, repeats=max_w, axis=2)
    freqs_2d = jnp.concatenate([freqs_h, freqs_w], axis=-1) # 1, H, W, d//2
    return freqs_2d.astype(jnp.complex64) # explicit



@jax.jit
def conv2d(model_params: PixtralModel, image_CHW: jax.Array) -> jax.Array:
    patch_size = 16 # params.json
    H, W = image_CHW.shape[1:]
    h, w = H//patch_size, W//patch_size
    patch_embeddings_HWO = jax.lax.conv_general_dilated(
        image_CHW[jnp.newaxis, ...],
        model_params.vision_encoder.patch_conv_weight,
        (patch_size, patch_size),
        'SAME',
        preferred_element_type=jnp.float32, # OPTIMIZATION: test in bfloat16 (do a grep on float32 actually...)
        precision=jax.lax.Precision.HIGHEST
    )
    return patch_embeddings_HWO.astype(jnp.bfloat16)




### Mask creating functions

def create_block_diagonal_mask(flattened_patch_embeddings_list: List[jax.Array]) -> jax.Array:
    patch_counts = [embeds.shape[0] for embeds in flattened_patch_embeddings_list]
    total_patch_count = sum(patch_counts)
    block_diagonal_mask = jnp.ones((total_patch_count, total_patch_count), dtype=bool)
    start_patch_idx = 0
    for patch_count in patch_counts:
        block_diagonal_mask = block_diagonal_mask.at[start_patch_idx:start_patch_idx+patch_count, start_patch_idx:start_patch_idx+patch_count].set(False)
        start_patch_idx = start_patch_idx + patch_count
    return block_diagonal_mask


def get_causal_mask(T: int) -> jax.Array:
    mask = jnp.ones((T, T), dtype=bool)
    mask = jnp.triu(mask, k=1)
    return mask




### Common math functions

@jax.jit
def layernorm(hidden_state_BTC: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    mean = jnp.mean(hidden_state_BTC, axis=-1, keepdims=True)
    std = jnp.std(hidden_state_BTC, axis=-1, keepdims=True)
    hidden_state_BTC = (hidden_state_BTC - mean)
    hidden_state_BTC = hidden_state_BTC/std
    hidden_state_BTC = hidden_state_BTC*weight # element-wise
    hidden_state_BTC = hidden_state_BTC + bias # element-wise
    return hidden_state_BTC



@jax.jit
def RMSnorm(hidden_state: jax.Array, weight: jax.Array) -> jax.Array:
    eps = 1e-5
    squared = jax.lax.pow(hidden_state, 2)
    mean = (jnp.mean(squared, axis=-1, keepdims=True) + eps)
    rsqrt = jax.lax.rsqrt(mean)
    hidden_state = jnp.multiply(hidden_state, rsqrt)
    hidden_state = jnp.multiply(hidden_state, weight)
    return hidden_state



def RMSnorm_lora(hidden_state: jax.Array, weight: jax.Array, lora_weight: jax.Array) -> jax.Array:
    eps = 1e-5
    squared = jax.lax.pow(hidden_state, 2)
    mean = (jnp.mean(squared, axis=-1, keepdims=True) + eps)
    rsqrt = jax.lax.rsqrt(mean)
    hidden_state = jnp.multiply(hidden_state, rsqrt)
    hidden_state = jnp.multiply(hidden_state, weight) + jnp.multiply(hidden_state, lora_weight)
    return hidden_state



def feed_forward(block_params: TransformerBlock, hidden_state_BTC: jax.Array) -> jax.Array:
    x = hidden_state_BTC
    x1 = jax.nn.silu(x @ block_params.feed_forward_w1_weight.T)
    x3 = x @ block_params.feed_forward_w3_weight.T
    x2 = (x1 * x3) @ block_params.feed_forward_w2_weight.T
    return x2



def feed_forward_lora(block_params: TransformerBlock, block_lora_params: LoRA, hidden_state_BTC: jax.Array) -> jax.Array:
    x = hidden_state_BTC

    x1 = x @ block_params.feed_forward_w1_weight.T
    x1 = x1 + ((x @ block_lora_params.block.ffw1_in) @ block_lora_params.block.ffw1_out)*block_lora_params.block.ffw1_alpha
    x1 = jax.nn.silu(x1)
    
    x3 = x @ block_params.feed_forward_w3_weight.T
    x3 = x3 + ((x @ block_lora_params.block.ffw3_in) @ block_lora_params.block.ffw3_out)*block_lora_params.block.ffw3_alpha

    x2 = (x1 * x3) @ block_params.feed_forward_w2_weight.T
    x2 = x2 + (((x1 * x3) @ block_lora_params.block.ffw2_in) @ block_lora_params.block.ffw2_out)*block_lora_params.block.ffw2_alpha
    
    return x2


