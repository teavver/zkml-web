import os, ezkl, json
from utils import PATHS
from torch import Tensor


async def ezkl_input_to_witness(input_path: str = PATHS["input"]):
    if not os.path.isfile(input_path):
        print(f"no input file found: ({input_path})")
        return None
    if os.path.isfile(PATHS["witness"]):
        os.remove(PATHS["witness"])
    await ezkl.gen_witness(input_path, PATHS["model_compiled"], PATHS["witness"])
    try:
        with open(PATHS["witness"], "r") as f:
            witness_data = json.load(f)
        return witness_data
    except Exception as e:
        print(e)
        return None


def tensor_to_ezkl_input(tensor: Tensor, dest_path: str = PATHS["input"]):
    if tensor.dim() != 4:
        print("invalid tensor shape")
        return None
    data_array = ((tensor).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data=[data_array])
    json.dump(data, open(dest_path, "w"))
