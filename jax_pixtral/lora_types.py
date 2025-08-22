#|==============================================================>
#|_______ lora_types.py _______________________________________
#|
#|
#|  Contains dataclasses for various loras
#| 
#|  Loras:
#|    - LoRA (all loras combined)
#|    - AttentionLoRA (Q, K, V, O)
#|    - MLPLora
#|    - DenseLoRA (lm head)
#|
#|
#|    Blog post on lora-from-scratch basics:
#|    https://ash-01xor.github.io/blog/posts/LoRA/
#|
#|
#|===============================================================>>>


import jax
import jax.numpy as jnp
from typing import NamedTuple, List



########################
## BlockLora
## lora for each xfmr block's RMSnorm1, RMSnorm2, and FFW
## layer-based SoA that will be scanned over

class BlockLoRA(NamedTuple):
    attnnorm : jax.Array # elementwise. not a lora
    ffwnorm: jax.Array # elementwise. not a lora
    ffw1_in: jax.Array
    ffw1_out: jax.Array
    ffw1_alpha: jax.Array # array of floats
    ffw2_in: jax.Array
    ffw2_out: jax.Array
    ffw2_alpha: jax.Array
    ffw3_in: jax.Array
    ffw3_out: jax.Array
    ffw3_alpha: jax.Array



#########################
## AttentionLora
## list of QLoRA, KLoRA, and VLoRA (QKV - NOT 'quantized')
## layer-based SoA. this will be scanned over

class AttentionLoRA(NamedTuple):
    in_q:  jax.Array
    out_q: jax.Array
    alpha_q: jax.Array 
    in_k:  jax.Array
    out_k: jax.Array
    alpha_k: jax.Array
    in_v:  jax.Array
    out_v: jax.Array
    alpha_v: jax.Array
    in_o:  jax.Array
    out_o: jax.Array
    alpha_o: jax.Array



################################
## DenseLora
## applies a lora to the lm head

class DenseLoRA(NamedTuple):
    in_matrix: jax.Array # (channel, lora_dim)
    out_matrix: jax.Array # (lora_dim, vocab)
    alpha: jnp.bfloat16



###########################
### LoRA
## all loras combined into one
## general-purpose lora type. used for function logic/signatures
## to disable specific sub-loras, set their ranks to 0 at initialization


# SoA to be scanned over. size (layers, ...)
class LoRALayers(NamedTuple):
    block: BlockLoRA
    attn: AttentionLoRA


class LoRA(NamedTuple):
    layers: LoRALayers
    dense: DenseLoRA



