from torch import nn
import os
import asyncio
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)

        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (32*26*26)
        print('shape pre flatten ', x.shape)
        x = x.flatten(start_dim = 1)
        print('shape post flatten :', x.shape)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        print('shape post d1 :', x.shape)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)

        return logits


model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.compiled')
pk_path = os.path.join('test.pk')
vk_path = os.path.join('test.vk')
settings_path = os.path.join('settings.json')
witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')


async def main():
    print('start')
    circuit = Net()
    shape = [1, 28, 28]
    # x = 0.1*torch.rand(1,*shape, requires_grad=True)
    x = 0.1*torch.rand(1,*shape, requires_grad=False)
    print(x.shape)
    circuit.eval()
    res = circuit(x)
    print(res)
    # torch.onnx.export(circuit,               # model being run
    #                     x,                   # model input (or a tuple for multiple inputs)
    #                     model_path,            # where to save the model (can be a file or file-like object)
    #                     export_params=True,        # store the trained parameter weights inside the model file
    #                     opset_version=10,          # the ONNX version to export the model to
    #                     do_constant_folding=True,  # whether to execute constant folding for optimization
    #                     input_names = ['input'],   # the model's input names
    #                     output_names = ['output'], # the model's output names
    #                     dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
    #                                     'output' : {0 : 'batch_size'}})

    # data_array = ((x).detach().numpy()).reshape([-1]).tolist()
    # data = dict(input_data = [data_array])
    # json.dump( data, open(data_path, 'w' ))
    
    
    
if __name__ == '__main__':
    print(123)
    asyncio.run(main())

