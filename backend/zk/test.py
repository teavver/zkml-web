import os, sys, torch
from utils import PATHS, local_img_to_b64, b64_to_tensor, show_tensor
from model.model import Net

IMGS_DIR = './test_imgs'
IMGS_EXT = '.png'

# Run test imgs
net = Net()
if not os.path.isfile(PATHS['model']):
    print(f'Model {PATHS['model']} not found')
    sys.exit(1)

net.load_state_dict(torch.load(PATHS["model"]))

for file in os.listdir(IMGS_DIR):
    if file.endswith(IMGS_EXT):
        b64 = local_img_to_b64(os.path.join(IMGS_DIR, file))
        tensor = b64_to_tensor(b64, True)
        # show_tensor(tensor)
        pred = net.predict(tensor)
        print(f'File: "{file}"'.ljust(24), f'Prediction: {pred}')
        