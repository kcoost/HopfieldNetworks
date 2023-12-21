from pathlib import Path
from datasets import load_dataset
import textwrap
from PIL import Image, ImageDraw, ImageFont

def draw_multiple_line_text(text, id):
    '''
    From unutbu on [python PIL draw multiline text on image](https://stackoverflow.com/a/7698300/395857)
    '''
    image = Image.new('1', (200, 100), color = (255,))
    fontsize = 10  # starting font size
    font = ImageFont.truetype("arial.ttf", fontsize)
    text_color = (0,)

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    y_text = 0
    lines = textwrap.wrap(text, width=40)
    for line in lines[:8]:
        line_width, line_height = font.getsize(line)
        draw.text(((image_width - line_width) / 2, y_text),
                line, font=font, fill=text_color)
        y_text += 12#line_height

    image.save(Path(__file__).parent / "texts" / f'text_{id}.png')



if __name__ == "__main__":
    n_max = 50
    dataset = load_dataset("wikipedia", "20220301.de", streaming=True, cache_dir=Path(__file__).parent / "cache")
    n = 0
    for x in dataset["train"]:
        if n == n_max:
            break
        draw_multiple_line_text(x["text"], x["id"])
        n += 1
