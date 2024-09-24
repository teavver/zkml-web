import base64, io
from PIL import Image
from torchvision import transforms

def local_img_to_b64(fname: str):
    try:
        ext = fname.split(".")[-1]
        prefix = f"data:image/{ext};base64,"
        with open(fname, "rb") as f:
            img = f.read()
        return prefix + base64.b64encode(img).decode("utf-8")
    except Exception as e:
        print('exc ', e)
    
def b64_to_tensor(b64: str):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    preprocess = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1 - x), # invert colors
        ]
    )
    tensor = preprocess(img)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor
