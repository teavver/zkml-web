import os, sys, torch
import matplotlib.pyplot as plt
from main import Net, PATHS, predict
from utils import local_img_to_b64, b64_to_tensor


IMGS_DIR = './test_imgs'
IMGS_EXT = '.png'

if __name__ == "__main__":
    # Run test imgs
    net = Net()
    if not os.path.isfile(PATHS['model']):
        print(f'Model {PATHS['model']} not found')
        sys.exit(1)
    
    net.load_state_dict(torch.load(PATHS["model"]))
        
    for file in os.listdir(IMGS_DIR):
        if file.endswith(IMGS_EXT):
            b64 = local_img_to_b64(os.path.join(IMGS_DIR, file))
            tensor = b64_to_tensor(b64)
            pred = predict(net, tensor)
            print(f'File: "{file}"'.ljust(24), f'Prediction: {pred}')
            