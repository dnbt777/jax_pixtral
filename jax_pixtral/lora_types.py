#|==============================================================>
#|_______ lora_types.py _______________________________________
#|
#|
#|  Contains dataclasses for various loras
#| 
#|  Loras:
#|    - LoRA (all loras combined)
#|    - AttentionLoRA (Q, K, V, O)
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



#########################
## AttentionLora
## list of QLoRA, KLoRA, and VLoRA (QKV - NOT 'quantized')
## layer-based SoA. this will be scanned over

class AttentionLoRALayer(NamedTuple):
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

class AttentionLoRA(NamedTuple):
    layers: AttentionLoRALayer



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

class LoRA(NamedTuple):
    attention_lora: AttentionLoRA
    dense_lora: DenseLoRA



