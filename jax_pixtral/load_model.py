#|=====================================>
#|_______ load_model.py _______________
#|
#|
#|  Loads pixtral from safetensors
#| 
#| 
#|======================>==============>>


import jax
import jax.numpy as jnp
from safetensors.numpy import load_file
from safetensors import safe_open
from jax_pixtral.model_types import * 
from jax.core import ShapedArray # for loading dummy tensors with no data 


# Print out the shapes of the model
# Alternatively, huggingface shows shapes in safetensor files:
# https://huggingface.co/mistralai/Pixtral-12B-2409/blob/main/consolidated.safetensors
def display_shapes(paths: str):
    for path in paths:
        tensors = load_file(path)
        print("loaded", path)
        for key, tensor in tensors.items():
            print(f"{key}: {tensor.shape}, {tensor.dtype}")
        del tensors 



# can likely be optimized
def load_params(paths: str) -> PixtralModel:
    """
    Loads pixtral's params
    """
    vision_encoder_layers = list(range(23+1)) # [0, 23]
    transformer_layers = list(range(39+1)) # [0, 39]
    model_dtype = "bfloat16"
    cpu = jax.devices("cpu")[0]

    path = paths[0] # temp (some models have multiple safetensors files. pixtral 12b only has one)

    with safe_open(path, framework="numpy") as f:
        load_tensor = lambda key: jax.device_put(f.get_tensor(key)) # straight to gpu

        def load_tensors(fmt_key, count): # stack on host, then send to gpu
            with jax.default_device(cpu):
                stacked = jnp.stack([f.get_tensor(fmt_key.format(i)) for i in range(count)])
            return jax.device_put(stacked)

        # load model
        model_params = PixtralModel(
            norm_weight=load_tensor(f"norm.weight"),
            output_weight=load_tensor(f"output.weight"),
            tok_embeddings_weight=load_tensor(f"tok_embeddings.weight"),
            vision_encoder=VisionEncoder(
              ln_pre_weight=load_tensor(f"vision_encoder.ln_pre.weight"),
              patch_conv_weight=load_tensor(f"vision_encoder.patch_conv.weight"),
              vision_encoder_layers=VisionEncoderLayer(
                  attention_wk_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wk.weight", count=len(vision_encoder_layers)),
                  attention_wo_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wo.weight", count=len(vision_encoder_layers)),
                  attention_wq_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wq.weight", count=len(vision_encoder_layers)),
                  attention_wv_weight=load_tensors("vision_encoder.transformer.layers.{}.attention.wv.weight", count=len(vision_encoder_layers)),
                  attention_norm_weight=load_tensors("vision_encoder.transformer.layers.{}.attention_norm.weight", count=len(vision_encoder_layers)),
                  feed_forward_w1_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w1.weight", count=len(vision_encoder_layers)),
                  feed_forward_w2_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w2.weight", count=len(vision_encoder_layers)),
                  feed_forward_w3_weight=load_tensors("vision_encoder.transformer.layers.{}.feed_forward.w3.weight", count=len(vision_encoder_layers)),
                  ffn_norm_weight=load_tensors("vision_encoder.transformer.layers.{}.ffn_norm.weight", count=len(vision_encoder_layers)),
              )
            ),
            vision_language_adapter=VisionLanguageAdapter(
              w_in_bias=load_tensor(f"vision_language_adapter.w_in.bias"),
              w_in_weight=load_tensor(f"vision_language_adapter.w_in.weight"),
              w_out_bias=load_tensor(f"vision_language_adapter.w_out.bias"),
              w_out_weight=load_tensor(f"vision_language_adapter.w_out.weight"),
            ),
            transformer=Transformer(
              transformer_layers=TransformerLayer(
                  attention_wk_weight=load_tensors("layers.{}.attention.wk.weight", count=len(transformer_layers)),
                  attention_wo_weight=load_tensors("layers.{}.attention.wo.weight", count=len(transformer_layers)),
                  attention_wq_weight=load_tensors("layers.{}.attention.wq.weight", count=len(transformer_layers)),
                  attention_wv_weight=load_tensors("layers.{}.attention.wv.weight", count=len(transformer_layers)),
                  attention_norm_weight=load_tensors("layers.{}.attention_norm.weight", count=len(transformer_layers)),
                  feed_forward_w1_weight=load_tensors("layers.{}.feed_forward.w1.weight", count=len(transformer_layers)),
                  feed_forward_w2_weight=load_tensors("layers.{}.feed_forward.w2.weight", count=len(transformer_layers)),
                  feed_forward_w3_weight=load_tensors("layers.{}.feed_forward.w3.weight", count=len(transformer_layers)),
                  ffn_norm_weight=load_tensors("layers.{}.ffn_norm.weight", count=len(transformer_layers)),
              )
            ),
        )
    
    return model_params



fast_load_params = load_params # legacy



