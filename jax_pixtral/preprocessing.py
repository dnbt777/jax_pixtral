#|==============================================================>
#|_______ preprocessing.py ______________________________________
#|
#|
#|  Preprocessing functions
#|    - Image processing
#|    - Tokenization
#|
#|
#|===============================================================>>>


import jax
import jax.numpy as jnp
from einops import rearrange
import jax.random as jrand
from functools import partial

from PIL import Image

import json
#import re
import regex as re
import base64

from typing import List, Tuple
from jax_pixtral.model_types import *



### Image processing functions

def convert_to_rgb(image: Image) -> Image:
    image = image.convert("RGBA")
    bg = Image.new("RGBA", image.size, "WHITE")
    bg.paste(image, (0, 0), image)
    rgb_img = bg.convert("RGB")
    return rgb_img



def process_image(image: Image) -> Tuple[jax.Array, Image]:
    ## get embeddings of img
    # convert to rgb
    image = convert_to_rgb(image)
    # resize img
    max_image_size = 1024 # from params.json
    patch_size = 16 # from params.json
    width, height = image.size
    resize_factor = max(height / max_image_size, width / max_image_size)
    if resize_factor > 1:
        height = round(height / resize_factor)
        width = round(width / resize_factor)
    height_tokens = int(jnp.ceil(height / patch_size))
    width_tokens = int(jnp.ceil(width / patch_size))
    
    new_size = (width_tokens*patch_size, height_tokens*patch_size) # no padding! nice!
    #image = cv2.resize(np.array(image, dtype=np.float32), new_size, interpolation=cv2.INTER_CUBIC) # mistral's version
    image = jnp.array(image.resize(new_size, resample=Image.BICUBIC), dtype=jnp.float32)
    
    # rescale/normalize
    # https://github.com/mistralai/mistral-common/blob/9a38768468fe012aac04bea4d3c33fdd0dd1fd59/src/mistral_common/tokens/tokenizers/multimodal.py#L67
    DATASET_MEAN = jnp.array((0.48145466, 0.4578275, 0.40821073), dtype=jnp.float32)
    DATASET_STD = jnp.array((0.26862954, 0.26130258, 0.27577711), dtype=jnp.float32)
    image = image / jnp.float32(255.0)
    image = (image - DATASET_MEAN) / DATASET_STD
    
    # CHANNEL_FIRST format
    image = jnp.array(image, dtype=jnp.bfloat16) # explicitly convert to jnp array w dtype float32. when x64 is enabled, implicit => float64 and breaks this
    image = jnp.transpose(image, (2, 0, 1))
    
    # create tokens
    img_token_id = 10 # from params.json
    img_break_id = 12 # from params.json
    img_end_id = 13 # from params.json
    image_tokens = [[img_token_id for _ in range(width_tokens)] + [img_break_id] for _ in range(height_tokens)]
    image_tokens = [item for sublist in image_tokens for item in sublist] # flatten img token list. 2d->1d. also wtf python weird ass syntax
    image_tokens[-1] = img_end_id # replace final row's break token with img end token
    
    return (image_tokens, image)



### Tokenization functions

"""
# https://docs.mistral.ai/guides/tokenization/
# mistral's tokenizer
import json
from mistral_common.tokens.tokenizers.tekken import Tekkenizer 
tok = Tekkenizer.from_file("./pixtral/tekken.json")

def encode(string, add_special=False):
    return tok.encode(string, bos=add_special, eos=add_special)

def decode(ids):
    return tok.decode(ids)
"""



def _bpe_encode_bytes(tokenizer, b):
    # rank correlates to token_id
    # rank = token_id - 1000 (specials)
    # this preserves order, though, so we can just use token_id in place of rank
    tokens = [bytes([x]) for x in b]
    while True:
        pairs = {
            (i, tokenizer["bytes_to_ids"][tokens[i] + tokens[i+1]], tokens[i]+tokens[i+1])
            for i in range(len(tokens)-1) if tokens[i] + tokens[i+1] in tokenizer["bytes_to_ids"].keys()
        }
        if not pairs:
            break
        i, _, merged = min(pairs, key=lambda x: x[1])
        tokens[i:i+2] = [merged]
    return [tokenizer["bytes_to_ids"][t] for t in tokens]



def _encode(tokenizer, string):
    ids = []
    for piece in tokenizer["regex"].findall(string):
        ids += _bpe_encode_bytes(tokenizer, piece.encode("utf-8"))
    return ids



def _decode(tokenizer, ids, keep_special=True):
    byte_chunks = []
    for i in ids:
        if i < tokenizer["special_token_count"] and not keep_special:
            continue
        if i >= tokenizer["special_token_count"]:
            byte_chunks.append(tokenizer["ids_to_bytes"][i])
        else:
            byte_chunks.append(tokenizer["special_tokens"][i].encode())
    return b"".join(byte_chunks).decode("utf-8", errors="replace")



def load_tokenizer(
    config_dir: str = None,
) -> dict:
    # handle default config
    if not config_dir:
        config_dir = "./pixtral"
    if config_dir[-1] != "/":
        config_dir = config_dir + "/"
    config_path = config_dir + "tekken.json"
    special_tokens_config_path = config_dir + "special_tokens.json"
    # load tekken config (comes with pixtral)
    with open(config_path, 'rb') as config_file:
        tekken_config = json.load(config_file)
    vocab = tekken_config["vocab"]
    regex_pattern = tekken_config["config"]["pattern"]
    special_token_count = tekken_config["config"]["default_num_special_tokens"]

    # load special tokens config (had to make this. not sure where it is on the internet)
    with open(special_tokens_config_path, 'r') as specials_config_file:
        special_tokens_config = json.load(specials_config_file)
    special_tokens = special_tokens_config["vocab"]

    # ids -> token string
    special_tokens = [(token["rank"], token["token_str"]) for token in special_tokens]
    special_tokens = sorted(special_tokens, key=lambda kv: kv[0])
    special_tokens = [token_str for _, token_str in special_tokens]
    while len(special_tokens) < special_token_count:
        padding = {"rank": len(special_tokens), "token_str": f"<err{len(special_tokens)}>", "is_control": True}
        special_tokens.append(padding)
    assert len(special_tokens) == special_token_count

    # id -> token string
    ids_to_bytes = [( token["rank"],
                      base64.b64decode(token["token_bytes"]),
                    ) for token in vocab]
    ids_to_bytes = sorted(ids_to_bytes, key=lambda kv: kv[0])
    ids_to_bytes = [token_bytes for _, token_bytes in ids_to_bytes]
    ids_to_bytes = special_tokens + ids_to_bytes

    # token bytes -> id
    # does not include specials
    vocab_size = tekken_config["config"]["default_vocab_size"]
    bytes_to_ids = {
        base64.b64decode(token["token_bytes"]): token["rank"] + special_token_count
        for token in vocab if token["rank"] + special_token_count < vocab_size
        # limit size to runtime max
    }

    tokenizer = {
        "ids_to_bytes": ids_to_bytes,
        "bytes_to_ids": bytes_to_ids,
        "special_tokens": special_tokens,
        "regex": re.compile(regex_pattern, re.UNICODE),
        "special_token_count": special_token_count,
        "MAX_ID": special_token_count + vocab_size
    }

    # create and return the encoder and decoder functions for this tokenizer
    encode = partial(_encode, tokenizer)
    decode = partial(_decode, tokenizer)
    
    return tokenizer, encode, decode



# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def tokenize_messages_dict(tokenizer, messages, add_eos=True):
    # create partial functions that already incorporate tokenizer param
    encode = partial(_encode, tokenizer)
    decode = partial(_decode, tokenizer)
    
    # token IDS https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
    BOS = 1 # <s>
    INS_START = 3 # [INS]
    INS_END = 4 # [/INS]
    EOS = 2 # </s>

    tokens = [BOS]
    images = []
    image_start_indices = []
    previous_role = None
    for message in messages:
        if message["role"] == "user":
            if previous_role != "user":
                    tokens.append(INS_START)
            if type(message["content"]) == str:
                tokens = tokens + encode(message["content"])
            else:
                for content in message["content"]:
                    if content["type"] == "text":
                        tokens = tokens + encode(content["text"])
                    elif content["type"] == "image_url":
                        image_start_index = len(tokens)
                        image_start_indices.append(image_start_index)
                        source = content["image_url"]["url"]
                        if ('https://' in source) or ('http://' in source):
                            pass # requests.get
                        elif 'data:image' in source:
                            pass # base64
                        else:
                            # file
                            image = Image.open(source)
                            image_tokens, processed_image = process_image(image)
                            tokens = tokens + image_tokens
                            images.append(processed_image)
                    else:
                        raise NameError(f"Error processing messages. Unknown content type {content["type"]}")
            previous_role = "user"
            # https://huggingface.co/mistral-community/pixtral-12b
            # append INS_END at end of user prompt
            # see 'usage example' in the link above
        elif message["role"] == "assistant":
            if previous_role == "user":
                    tokens.append(INS_END)
            assert len(message["content"]) == 1 # assistant will only produce 1 response
            tokens = tokens + encode(message["content"][0]["text"])
            if add_eos:
                    tokens = tokens + [EOS]
            previous_role = "assistant"
        else:
            raise NameError(f"Error processing messages. Unknown role {message["role"]}")
    if previous_role == "user":
            tokens.append(INS_END)
    return tokens, images, image_start_indices



# example message format: https://docs.vllm.ai/en/v0.8.0/getting_started/examples/pixtral.html
def tokenize_messages_dict_with_masks(tokenizer, messages, add_eos=True):
    """
    context mask: masks out everything except the assistant's response. used for fine-tuning
    image_mask: masks out image tokens. used for full training
    """
    # create partial functions that already incorporate tokenizer param
    encode = partial(_encode, tokenizer)
    decode = partial(_decode, tokenizer)
    
    # token IDS https://github.com/mistralai/mistral-common/issues/105#issuecomment-2997200779
    BOS = 1 # <s>
    INS_START = 3 # [INS]
    INS_END = 4 # [/INS]
    EOS = 2 # </s>

    tokens = []
    images = []
    image_start_indices = []
    image_mask = []
    context_mask = []
    previous_role = None

    # append initial <s>
    tokens.append(BOS)
    context_mask.append((True, 1)) # hide these tokens
    image_mask.append((False, 1)) # dont hide these tokens
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if previous_role != "user":
                    tokens.append(INS_START)
                    context_mask.append((True, 1)) # hide these tokens
                    image_mask.append((False, 1)) # dont hide these tokens
            if type(message["content"]) == str:
                new_tokens = encode(message["content"])
                tokens = tokens + new_tokens
                context_mask.append((True, len(new_tokens)))
                image_mask.append((False, len(new_tokens)))
            else:
                for content in message["content"]:
                    if content["type"] == "text":
                        new_tokens = encode(content["text"])
                        tokens = tokens + new_tokens
                        context_mask.append((True, len(new_tokens)))
                        image_mask.append((False, len(new_tokens)))
                    elif content["type"] == "image_url":
                        image_start_index = len(tokens)
                        image_start_indices.append(image_start_index)
                        source = content["image_url"]["url"]
                        if ('https://' in source) or ('http://' in source):
                            pass # requests.get
                        elif 'data:image' in source:
                            pass # base64
                        else:
                            # file
                            image = Image.open(source)
                            image_tokens, processed_image = process_image(image)
                            tokens = tokens + image_tokens
                            images.append(processed_image)
                            context_mask.append((True, len(image_tokens)))
                            image_mask.append((True, len(image_tokens)))
                    else:
                        raise NameError(f"Error processing messages. Unknown content type {content["type"]}")
            previous_role = "user"
            # https://huggingface.co/mistral-community/pixtral-12b
            # append INS_END at end of user prompt
            # see 'usage example' in the link above
        elif message["role"] == "assistant":
            if previous_role == "user":
                tokens.append(INS_END)
                context_mask.append((True, 1))
                image_mask.append((False, 1))
            assert len(message["content"]) == 1 # assistant should only produce 1 response
            new_tokens = encode(message["content"][0]["text"])
            if add_eos:
                    new_tokens = new_tokens + [EOS]
            tokens = tokens + new_tokens
            is_final_response = i == len(messages) - 1
            context_mask.append((not is_final_response, len(new_tokens)))
            image_mask.append((False, len(new_tokens)))
            previous_role = "assistant"
        else:
            raise NameError(f"Error processing messages. Unknown role {message["role"]}")
    if previous_role == "user":
            tokens.append(INS_END)
            context_mask.append((True, 1))
            image_mask.append((False, 1))
    # unpack context_mask and image_masks
    # [(x, 2), (y, 3), (z, 4)] => [x, x, y, y, y, z, z, z, z]
    def unpack_list(packed_list):
        unpacked_list = [[val for _ in range(count)] for val, count in packed_list]
        unpacked_list = [item for sublist in unpacked_list for item in sublist]
        return unpacked_list

    context_mask = unpack_list(context_mask)
    image_mask = unpack_list(image_mask)
        
    return tokens, images, image_start_indices, context_mask, image_mask

