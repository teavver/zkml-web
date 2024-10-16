import base64, io, torch, os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import transforms
from model.model import Net
from typing import Union
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection


PATHS = {
    "model": "mnist.pt",
    "model_compiled": "mnist.compiled",
    "model_onnx": "network.onnx",
    # EZKL
    ## temp
    "input": "input.json",
    "witness": "witness.json",
    "proof": "proof.pf",
    ## one-time setup
    "settings": "settings.json",
    "pk": "prover_key.pk",
    "vk": "verifier_key.vk",
    "calibration": "calibration.json",
    "srs": "srs",
    ## /verify user custom
    "vk_user": "vk_user",
    "srs_user": "srs_user",
}


def env_check():
    expected_keys = ["DB_USER", "DB_PASS"]
    for key in expected_keys:
        val = os.getenv(key)
        if val is None:
            print(f'key "{key} is missing in .env, app might crash')


def read_file(fname: str) -> str | None:
    try:
        with open(fname) as f:
            c = f.read()
        return c
    except Exception as e:
        print(f"failed to read file {fname}, err: {e}")
        return None


def write_file(content: Union[str, bytes], fname: str):
    try:
        mode = "w" if isinstance(content, str) else "wb"
        with open(fname, mode) as f:
            f.write(content)
    except Exception as e:
        print(f"failed to write file {fname}, err: {e}")


def create_db_client() -> Collection:
    uri = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@cluster0.e48xh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri, server_api=ServerApi("1"))
    collection = None
    try:
        client.admin.command("ping")
        db = client.get_database("zkml_web")
        collection = db["records"]
    except Exception as e:
        print(e)
    return collection


def load_net():
    net = Net()
    net.load_state_dict(torch.load(PATHS["model"]))
    return net


def parse_b64(b64data: str):
    try:
        B64_PREFIX = "base64,"
        b64data = "".join(b64data.strip().split())
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
        print("exc ", e)


def b64_to_tensor(b64: str, blur=False, invert=True):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if invert:
        img = ImageOps.invert(img)
    preprocess = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    if blur:
        preprocess.transforms.insert(
            -1, transforms.GaussianBlur(kernel_size=5, sigma=(0.04, 1.0))
        )
    tensor = preprocess(img)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def show_tensor(tensor, title=None):
    # display [1, 1, 28, 28] input tensor
    img = tensor.squeeze(0).squeeze(0)
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
