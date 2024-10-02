from time import time
from flask import Flask, request, current_app, jsonify
from utils import parse_b64, b64_to_tensor, load_net, read_file
from zk.ezkl_utils import tensor_to_ezkl_input, ezkl_input_to_witness
from zk.zk import ezkl_prove
from utils import create_db_client
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class PredictionResult:
    input: str # b64
    prediction_res: int
    timestamp: int
    id: int
    

@dataclass(frozen=True)
class PredictionRecord(PredictionResult):
    proof: str


def create_app():
    
    app = Flask(__name__)
    
    with app.app_context():
        current_app.net = load_net()
        current_app.db = create_db_client()
        print('model init ok')
    
    @app.route('/', methods=['GET'])
    def hello_world():
        return '<p>Hello, World!</p>'

    @app.route('/predict', methods=['POST'])
    async def predict():
        ts = int(time())
        request_data = request.get_json()
        body_input_key = 'input'
        input_img = request_data.get(body_input_key, None)
        if input_img is None or not isinstance(input_img, str):
            return f'invalid body data, expected key: "{body_input_key}"', 400
        try:
            input_img.encode('utf-8')
        except UnicodeEncodeError:
            return 'utf8 only bro', 400
        parsed_b64 = parse_b64(input_img)
        if parsed_b64 is None:
            return 'invalid b64 data', 400
        tensor = b64_to_tensor(parsed_b64)
        pred_res = current_app.net.predict(tensor)

        tensor_to_ezkl_input(tensor)
        await ezkl_input_to_witness()
        
        id = current_app.db.count_documents({}) + 1
        
        proof_path = f"proof.pf" 
        p = ezkl_prove(proof_path)
        if p == False:
            return 'internal err (proof)', 500
        
        res = PredictionResult(
            parsed_b64,
            pred_res,
            ts,
            id
        ).__dict__
        
        proof = read_file(proof_path)
        record = PredictionRecord(**res, proof=proof).__dict__
        
        db_res = current_app.db.insert_one(record)
        if db_res.acknowledged != True:
            return 'internal err (db insert)', 500
        
        return res, 200
    
    @app.route('/proof', methods=['GET'])
    async def get_proof():
        proof_id = request.args.get('proof_id', default=None, type=str)
        print(proof_id)
        return 'ok', 200

    return app