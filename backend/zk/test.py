import os, sys, torch, asyncio
from utils import PATHS, local_img_to_b64, b64_to_tensor, show_tensor
from zk.ezkl_utils import tensor_to_ezkl_input, ezkl_input_to_witness
from zk.zk import ezkl_full_setup, ezkl_verify, ezkl_prove
from model.model import Net

IMGS_DIR = './test_imgs'
IMGS_EXT = '.png'

# Test model on sample images
async def test_model():
    net = Net()
    if not os.path.isfile(PATHS['model']):
        print(f"Model {PATHS['model']} not found")
        sys.exit(1)

    net.load_state_dict(torch.load(PATHS["model"]))

    for file in os.listdir(IMGS_DIR):
        if file.endswith(IMGS_EXT):
            b64 = local_img_to_b64(os.path.join(IMGS_DIR, file))
            tensor = b64_to_tensor(b64, True)
            # show_tensor(tensor)
            pred = net.predict(tensor)
            print(f'File: "{file}"'.ljust(24), f'Prediction: {pred}')
        

# Test if zkml was setup correctly
async def test_ezkl():
    
    if not os.path.isfile(PATHS["pk"]) or not os.path.isfile(PATHS["vk"]):
        await ezkl_full_setup()
    
    x = 0.1 * torch.rand(1, *[1, 28, 28], requires_grad=True)
    tensor_to_ezkl_input(x) # input.json
    await ezkl_input_to_witness() # witness.json
    
    proof_ok = ezkl_prove()
    assert proof_ok == True
    
    verify_ok = ezkl_verify()
    assert verify_ok == True
    
    print('ezkl test OK')
    
    
if __name__ == "__main__":
    asyncio.run(test_model())
    asyncio.run(test_ezkl())