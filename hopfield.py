from pathlib import Path
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from tqdm import tqdm

def image_to_x(image):
    image = image.to(dtype=torch.int16) // 255
    image = image.flatten()
    image = 2*image - 1
    return image.unsqueeze(1)

def mask_image(image):
    image[:, 12:] = 255
    # save_image(image/255, Path(__file__).parent / "test.png")
    return image

def retreive(x, W):
    E = 1e9
    while True:
        x = torch.sign(W @ x)
        E_new = -0.5 * (x.T @ W @ x).item()
        if E <= E_new:
            break
        else:
            E = E_new
    return x

def get_W(dir):
    image_paths = list(dir.glob("*.png"))
    W = 0

    for image_path in tqdm(image_paths):
        image = read_image(str(image_path), mode = ImageReadMode.GRAY)
        x = image_to_x(image)
        W += x * x.T
    return W

if __name__ == "__main__":
    texts_dir = (Path(__file__).parent / "texts")
    W = get_W(texts_dir)

    image_paths = list(texts_dir.glob("*.png"))
    for image_path in tqdm(image_paths):
        image = read_image(str(image_path), mode = ImageReadMode.GRAY)
        masked_image = mask_image(image)
        x = image_to_x(masked_image)
        x_retreived = retreive(x, W)
        x_orig = image_to_x(image)
        print((x_retreived == x_orig).sum() / x_retreived.numel())
    print(2)