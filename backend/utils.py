import base64, io, torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
from model.model import Net
from ezkl.ezkl import PATHS

def load_net():
    net = Net()
    net.load_state_dict(torch.load(PATHS["model"]))
    return net

def parse_b64(b64data: str):
    try:
        B64_PREFIX = 'base64,'
        b64data = ''.join(b64data.strip().split())
        if B64_PREFIX in b64data:
            b64data = b64data.split(B64_PREFIX)[-1]
        base64.b64decode(b64data, validate=True)
        return b64data
    except Exception as e:
        print(e)
        return None

def local_img_to_b64(fname: str):
    try:
        ext = fname.split(".")[-1]
        prefix = f"data:image/{ext};base64,"
        with open(fname, "rb") as f:
            img = f.read()
        return prefix + base64.b64encode(img).decode("utf-8")
    except Exception as e:
        print('exc ', e)
    
def b64_to_tensor(b64: str, blur=False, invert=True):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert('RGB')
    if invert: img = ImageOps.invert(img)
    preprocess = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    if blur: preprocess.transforms.insert(-1, transforms.GaussianBlur(kernel_size=5, sigma=(0.04, 1.0)))
    tensor = preprocess(img)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

def show_tensor(tensor, title=None):
    # display [1, 1, 28, 28] input tensor
    img = tensor.squeeze(0).squeeze(0)
    plt.imshow(img, cmap='gray')
    if title: plt.title(title)
    plt.axis('off')
    plt.show()