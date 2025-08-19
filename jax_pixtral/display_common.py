#|==============================================================>
#|_______ display_common.py _____________________________________
#|
#|
#|  Functions/values for prettyprinting text/images
#|
#|  Values
#|    - Standard colors
#|    - Jax colors
#| 
#|  Functions
#|    - Add highlight/font colors to strings
#|    - Color printing functions
#|    - Terminal image display functions
#|
#|  Conventions:
#|    - Color: tuple(r, g, b)
#|    ~ Colors with <100% opacity are treated as transparent
#|
#|
#|===============================================================>>>


from PIL import Image
import jax.numpy as jnp
from typing import Optional, Tuple, TypeAlias



### define colors
Color: TypeAlias = Tuple[int, int, int]
# standard
color_black = (0, 0, 0)
color_grey = (130, 130, 130)
color_red = (255, 0, 0)
color_green = (0, 255, 0)
color_blue = (0, 0, 255)
# jax
color_jax_purple_light = (234, 128, 252)
color_jax_purple_med = (0, 105, 92)
color_jax_purple_dark = (106, 27, 154)
color_jax_blue_light = (94, 151, 246)
color_jax_blue_dark = (42, 86, 198)
color_jax_green_light = (38, 166, 154)
color_jax_green_dark = (0, 105, 92)



# Add highlight/font colors to strings

def color_string(s: str, color: Color) -> str:
    """ returns $string with $color font-color """
    r, g, b = color
    return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"



def highlight_string(s: str, color: Color) -> str:
    """ returns $string highlited with $color """
    r, g, b = color
    return f"\x1b[48;2;{r};{g};{b}m{s}\x1b[0m"



# Color printing functions

def print_color(*msgs, color: Color=color_black, end="\n"):
    """ same as print(strings), but with colors (same api) """
    if len(color) == 4:
        r, g, b, a = color
        if a < 255:
            print(" ", end=end, flush=True)
            return
    else:
        r, g, b = color
    msg = " ".join([str(part) for part in msgs])
    print(f"\x1b[38;2;{r};{g};{b}m{msg}\x1b[0m", end=end, flush=True)



def set_text_color(color: Color):
    """ changes the font-color of future printed characters """
    r, g, b = color
    print(f"\x1b[38;2;{r};{g};{b}m", end="", flush=True)



def reset_text_color():
    """ resets the font-color to default """
    print("\x1b[0m", end="", flush=True)



# Terminal image display functions

def two_pixel_column(top_pixel_color: Color, bottom_pixel_color: Color) -> str:
    """ create a colored char that looks like a top and bottom pixel """
    if len(top_pixel_color) == 4:
        r0, g0, b0, a0 = top_pixel_color
    else:
        r0, g0, b0 = top_pixel_color

    if len(bottom_pixel_color) == 4:
        r1, g1, b1, a1 = bottom_pixel_color
    else:
        r1, g1, b1 = bottom_pixel_color

    if (len(top_pixel_color) == 4 and len(bottom_pixel_color) == 4):
        if (a0 < 255) and (a1 < 255):
            return " " # if any transparency, treat as completely transparent
    
    RESET = "\x1b[0m"
    upper_half_block = "▀"
    font_format_string = f"\x1b[38;2;{int(r0)};{int(g0)};{int(b0)}m"
    background_format_string = f"\x1b[48;2;{int(r1)};{int(g1)};{int(b1)}m"
    return font_format_string + background_format_string + upper_half_block + RESET



def show_image(
    image: Image,
    height: int = 32,
    resample: Optional[Image.Resampling] = None
):
    """ prints $image to the terminal using two-pixel columns """
    resize_ratio = height / image.height
    new_size = (int(image.width*resize_ratio), height)
    if resample:
        image = image.resize(new_size, resample=resample)
    else:
        image = image.resize(new_size) # no resampling - keep cool pixellated look?
    image = jnp.array(image)

    for j in range(0, image.shape[0], 2):
        for i in range(0, image.shape[1], 1):
            top_i, top_j = i, j
            bottom_i, bottom_j = i, j+1
            top_pixel_color = image[top_j, top_i]
            bottom_pixel_color = image[bottom_j, bottom_i]
            print(two_pixel_column(top_pixel_color, bottom_pixel_color), end="", flush=True)
        print()



def load_and_show_image(
    image_url: str,
    height: int = 32,
    resample: Optional[Image.Resampling] = None
):
    """ loads an image from $image_url and shows it using show_image """
    # todo add base64 and url downloads (maybe..)
    image = Image.open(image_url)
    show_image(image, resample=resample)




