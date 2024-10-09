import argparse, torch, os, ezkl, json, asyncio
from time import time
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from .ezkl_utils import (
    ezkl_input_to_witness,
    tensor_to_ezkl_input,
)
from model.model import Net
from utils import PATHS

# Most of the code here comes from this example notebook:
# https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_network_output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), PATHS["model"])


# EZKL


def export_to_onnx(model: Net):
    model.eval()
    x = 0.1 * torch.rand(1, *[1, 28, 28], requires_grad=True)
    model_path = os.path.join(PATHS["model_onnx"])
    torch.onnx.export(
        model,
        x,
        model_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    tensor_to_ezkl_input(x)


async def ezkl_clear_config():
    for path in PATHS:
        if not "model" in path:
            if "calibration" in PATHS[path]:
                continue
            if os.path.isfile(PATHS[path]):
                os.remove(PATHS[path])
                print(f"File {PATHS[path]} removed")


def ezkl_prove(proof_path: str = PATHS["proof"]) -> bool:
    if os.path.isfile(proof_path):
        os.remove(proof_path)
    ezkl.prove(
        PATHS["witness"],
        PATHS["model_compiled"],
        PATHS["pk"],
        proof_path,
        "single",
        PATHS["srs"],
    )
    if not os.path.isfile(proof_path):
        print("ezkl failed to prove this computation")
        # todo: debug info about record
        return False
    return True


def ezkl_verify(proof_path: str = PATHS["proof"], vk_path: str = PATHS["vk"], srs_path: str = PATHS["srs"]):
    try:
        res = ezkl.verify(proof_path, PATHS["settings"], vk_path, srs_path)
        if res != True:
            print("ezkl failed to verify this computation")
            # todo: debug info about record
            return False
        return True
    except Exception as e:
        print(f"ezkl_verify exc: {e}")
        return False


async def ezkl_configure(model_onnx_path: str = PATHS["model_onnx"]):
    # first make sure model was exported to onnx and input.json is present
    assert os.path.isfile(PATHS["model_onnx"]), "No .onnx model found"
    assert os.path.isfile(
        PATHS["input"]
    ), "No input.json found. Did you run export_to_onnx?"

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"

    res = ezkl.gen_settings(model_onnx_path, PATHS["settings"], py_run_args=py_run_args)
    assert res == True  # Make sure we good before calibrating
    print("ezkl settings OK")

    if not os.path.isfile(PATHS["calibration"]):
        data_array = (
            (torch.rand(10, *[1, 28, 28], requires_grad=True).detach().numpy())
            .reshape([-1])
            .tolist()
        )
        data = dict(input_data=[data_array])
        json.dump(data, open(PATHS["calibration"], "w"))  # Dump calibration.json

        await ezkl.calibrate_settings(
            PATHS["calibration"], model_onnx_path, PATHS["settings"], "resources"
        )
        print("ezkl calibration OK")

    res = ezkl.compile_circuit(
        model_onnx_path, PATHS["model_compiled"], PATHS["settings"]
    )
    assert res == True
    print("ezkl circuit OK")

    res = await ezkl.get_srs(PATHS["settings"], None, PATHS["srs"])
    assert res == True
    assert os.path.isfile(PATHS["srs"])

    res = await ezkl_input_to_witness()
    assert os.path.isfile(PATHS["witness"])

    print("running ezkl setup...")

    res = ezkl.setup(
        PATHS["model_compiled"],
        PATHS["vk"],
        PATHS["pk"],
        PATHS["srs"],
        PATHS["witness"],
    )
    assert res == True
    assert os.path.isfile(PATHS["vk"])
    assert os.path.isfile(PATHS["pk"])
    assert os.path.isfile(PATHS["settings"])
    print("ezkl setup complete")


async def ezkl_full_setup(cleanup: bool = True):
    start = time()
    net = Net()
    
    if not os.path.isfile(PATHS["model"]):
        print(f'No model file, cant continue EZKL setup')
        os._exit(1)

    net.load_state_dict(torch.load(PATHS["model"]))
    if cleanup:
        await ezkl_clear_config()
    export_to_onnx(net)
    await ezkl_configure()
    end = time()
    print(f"Done in {int(end - start)}s")


if __name__ == "__main__":
    asyncio.run(ezkl_full_setup())
