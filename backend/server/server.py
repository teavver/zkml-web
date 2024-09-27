from flask import Flask, request, current_app
from utils import parse_b64, b64_to_tensor, load_net
from zk.ezkl_utils import tensor_to_ezkl_input, ezkl_input_to_witness
from dataclasses import dataclass

@dataclass
class PredictionRecord:
    input_img: str # b64
    prediction_res: int
    proof_id: int
    timestamp: int
    

def create_app():
    
    app = Flask(__name__)
    
    with app.app_context():
        current_app.net = load_net()
        print('model init ok')
    
    @app.route('/', methods=['GET'])
    def hello_world():
        return '<p>Hello, World!</p>'

    @app.route('/predict', methods=['POST'])
    async def handle_predict():
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
        pred = current_app.net.predict(tensor)
        
        test_id = 127
        test_path = f"prediction_{test_id}.json"
        tensor_to_ezkl_input(tensor, test_path)
        await ezkl_input_to_witness(test_path)
        return f'result: {pred}', 200

    return app