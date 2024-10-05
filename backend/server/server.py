import os
from typing import List
from time import time
from flask import Flask, request, current_app, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils import parse_b64, b64_to_tensor, load_net, read_file, write_file, show_tensor
from zk.ezkl_utils import tensor_to_ezkl_input, ezkl_input_to_witness
from zk.zk import ezkl_prove, ezkl_verify
from utils import create_db_client, PATHS
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PredictionResult:
    prediction_res: int
    timestamp: int
    id: int


@dataclass(frozen=True)
class PredictionRecord(PredictionResult):
    input: str  # b64
    proof: str


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512000  # 500KB limit
    CORS(app)

    with app.app_context():
        current_app.net = load_net()
        current_app.db = create_db_client()
        print("model init ok")

    @app.route("/", methods=["GET"])
    def root():
        return "ZKML-web api", 200

    @app.route("/get_records", methods=["GET"])
    async def get_records():
        limit = 10
        page_query = "page"
        page = request.args.get(page_query, default=0, type=int)
        if not isinstance(page, int):
            return (
                f'invalid body data, expected key: "{page_query}" type: int',
                400,
            )

        if page < 0:
            return "page number must be a positive integer", 400
        skip = page * limit
        cursor = (
            current_app.db.find(projection={"_id": 0, "proof": 0})
            .sort("_id", -1)
            .skip(skip)
            .limit(limit)
        )
        records = [PredictionRecord(**doc, proof="") for doc in cursor]
        return {"records": [asdict(record) for record in records]}, 200

    @app.route("/get_vk", methods=["GET"])
    async def get_vk():
        if not os.path.isfile(PATHS["vk"]):
            return f"we lost the vk file. GG", 500
        return send_file(os.path.abspath(PATHS["vk"]), as_attachment=True)
        

    @app.route("/get_proof", methods=["GET"])
    async def get_proof():
        proof_req_query = "id"
        req_id = request.args.get(proof_req_query, default=None, type=int)
        if req_id is None or not isinstance(req_id, int):
            return (
                f'invalid body data, expected key: "{proof_req_query}" type: int',
                400,
            )

        document = current_app.db.find_one({"id": req_id})
        if not document:
            return f"no record with id={req_id}", 404

        proof_data = document.get("proof", "no proof available")
        if os.path.isfile(PATHS["proof"]):
            os.remove(PATHS["proof"])

        write_file(proof_data, PATHS["proof"])
        return send_file(os.path.abspath(PATHS["proof"]), as_attachment=True, download_name=f'proof_{req_id}.pdf')

    @app.route("/predict", methods=["POST"])
    async def predict():
        ts = int(time())
        request_data = request.get_json()
        body_input_key = "input"
        input_img = request_data.get(body_input_key, None)
        if input_img is None or not isinstance(input_img, str):
            return f'invalid body data, expected key: "{body_input_key}" type: str', 400
        try:
            input_img.encode("utf-8")
        except UnicodeEncodeError:
            return "utf8 only bro", 400
        parsed_b64 = parse_b64(input_img)
        if parsed_b64 is None:
            return "invalid b64 data", 400
        tensor = b64_to_tensor(parsed_b64)
        pred_res = current_app.net.predict(tensor)
        tensor_to_ezkl_input(tensor)
        await ezkl_input_to_witness()

        next_id = current_app.db.count_documents({}) + 1
        proof_path = f"proof.pf"
        p = ezkl_prove(proof_path)
        if p == False:
            return "internal err (proof)", 500

        res = asdict(PredictionResult(pred_res, ts, next_id))

        proof = read_file(proof_path)
        if proof == None:
            return "proof file read err", 500

        record = asdict(PredictionRecord(**res, input=parsed_b64, proof=proof))
        db_res = current_app.db.insert_one(record)
        if db_res.acknowledged != True:
            return "internal err (db insert)", 500

        return jsonify(res)

    @app.route("/verify", methods=["POST"])
    async def verify_proof():
        if "file" not in request.files:
            return 'required "file" in req', 400

        file = request.files["file"]
        if file.filename == "":
            return "no selected file", 400

        fname = secure_filename(file.filename)
        if fname[len(fname) - 3 :] != ".pf":
            return "invalid file extension", 400

        if os.path.isfile(PATHS["proof"]):
            os.remove(PATHS["proof"])

        try:
            proof_data = file.read().decode("utf-8")
            write_file(proof_data, PATHS["proof"])
        except UnicodeDecodeError as e:
            print(f"/verify file encoding err: {e}")
            return "invalid file encoding", 400

        res = ezkl_verify(PATHS["proof"])
        if res == False:
            return "NOT OK", 200

        return "OK", 200

    return app
