import torch
from flask import Flask, request
from ..utils import parse_b64, b64_to_tensor
from ..model import Net, predict, PATHS

model = None

app = Flask(__name__)

def init_model():
    global model
    net = Net()
    net.load_state_dict(torch.load(PATHS["model"]))
    model = net


if __name__ == '__main__':
    model = init_model()
    app.run(debug=True)
    
@app.route('/', methods=['GET'])
def hello_world():
    return '<p>Hello, World!</p>'

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.get_json()
    data = request_data.get('data', None)
    if data is None or not isinstance(data, str):
        return 'invalid body data', 400
    try:
        data.encode('utf-8')
    except UnicodeEncodeError:
        return 'utf8 only bro', 400
    parsed_b64 = parse_b64(data)
    if parsed_b64 is None:
        return 'invalid b64 data', 400
    tensor = b64_to_tensor(parsed_b64)
    # res = predict(,tensor)
    # print(res)
    # return f'result: {res}', 200
