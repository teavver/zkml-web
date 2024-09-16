import torch, os, ezkl, json, asyncio
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output


async def main():
  model_path = os.path.join("network.onnx")
  compiled_model_path = os.path.join("network.compiled")
  pk_path = os.path.join("test.pk")
  vk_path = os.path.join("test.vk")
  settings_path = os.path.join("settings.json")
  witness_path = os.path.join("witness.json")
  data_path = os.path.join("input.json")
  print(model_path)

  circuit = Net()
  shape = [1, 28, 28]
  # After training, export to onnx (network.onnx) and create a data file (input.json)
  x = 0.1 * torch.rand(1, *shape, requires_grad=True)
  # Flips the neural net into inference mode
  circuit.eval()

  # Export the model
  print("onnx start")
  torch.onnx.export(
    circuit,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    model_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
      "input": {0: "batch_size"},  # variable length axes
      "output": {0: "batch_size"},
    },
  )

  data_array = ((x).detach().numpy()).reshape([-1]).tolist()
  data = dict(input_data=[data_array])
  # Serialize data into file:
  json.dump(data, open(data_path, "w"))

  py_run_args = ezkl.PyRunArgs()
  py_run_args.input_visibility = "private"
  py_run_args.output_visibility = "public"
  py_run_args.param_visibility = "fixed"  # private by default

  print("settings start")
  res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
  assert res == True
  print("settings")

  # cal_path = os.path.join("calibration.json")
  # print("calibration init")
  # data_array = (
  #   (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()
  # )
  # data = dict(input_data=[data_array])
  # # Serialize data into file:
  # json.dump(data, open(cal_path, "w"))

  # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources", 1)

  res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
  assert res == True

  res = await ezkl.get_srs(settings_path)

  res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
  assert os.path.isfile(witness_path)

  res = ezkl.setup(
    compiled_model_path,
    vk_path,
    pk_path,
  )

  assert res == True
  assert os.path.isfile(vk_path)
  assert os.path.isfile(pk_path)
  assert os.path.isfile(settings_path)


if __name__ == "__main__":
  asyncio.run(main())
