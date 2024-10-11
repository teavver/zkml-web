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


KB_IN_BYTES = 1024


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
    CORS(app)
    
    # accept big requests, but only for /verify
    @app.before_request
    def limit_content_length():
        if request.endpoint == "verify_proof":
            app.config["MAX_CONTENT_LENGTH"] = 6 * KB_IN_BYTES * KB_IN_BYTES  # 6MB limit
        else:
            app.config["MAX_CONTENT_LENGTH"] = 500 * KB_IN_BYTES # 500KB limit

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
            return f"we lost the vk file", 500
        return send_file(os.path.abspath(PATHS["vk"]), as_attachment=True)


    @app.route("/get_srs", methods=["GET"])
    async def get_srs():
        if not os.path.isfile(PATHS["srs"]):
            return f"we lost the srs file", 500
        return send_file(os.path.abspath(PATHS["srs"]), as_attachment=True)


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
        return send_file(
            os.path.abspath(PATHS["proof"]),
            as_attachment=True,
            download_name=f"proof_{req_id}.pf",
        )

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
            return "internal err (pf file read)", 500

        record = asdict(PredictionRecord(**res, input=parsed_b64, proof=proof))
        db_res = current_app.db.insert_one(record)
        if db_res.acknowledged != True:
            return "internal err (db insert)", 500

        return jsonify(res)

    @app.route("/verify", methods=["POST"])
    async def verify_proof():

        vk_path = PATHS["vk"]
        srs_path = PATHS["srs"]

        if "proof" not in request.files:
            return 'required "proof" in req', 400

        pf_file = request.files["proof"]
        if pf_file.filename == "":
            return "no selected proof file", 400

        fname = secure_filename(pf_file.filename)
        if fname[len(fname) - 3 :] != ".pf":
            return "invalid proof file extension", 400

        try:
            proof_data = pf_file.read().decode("utf-8")
            write_file(proof_data, PATHS["proof"])
        except UnicodeDecodeError as e:
            print(f"/verify proof file encoding err: {e}")
            return "invalid proof file encoding", 400

        # user VK
        if "vk" in request.files:
            vk_expected_fname = ".vk"
            vk_file = request.files["vk"]
            vk_fname = secure_filename(vk_file.filename)

            if vk_fname[len(vk_fname) - 3 :] != vk_expected_fname:
                return f"invalid VK file extension. '{vk_expected_fname}' expected.", 400

            try:
                vk_data = vk_file.read()
                if isinstance(vk_data, bytes):
                    write_file(vk_data, PATHS["vk_user"])
                    vk_path = PATHS["vk_user"]
                else:
                    return "error while reading VK file", 400
            except Exception as e:
                print(f"/verify VK file encoding err: {e}")
                return "invalid VK file format", 400


        # user SRS
        if "srs" in request.files:
            srs_file = request.files["srs"]
            if srs_file.filename == "":
                return "no selected SRS file", 400

            try:
                srs_data = srs_file.read()
                if isinstance(srs_data, bytes):
                    write_file(srs_data, PATHS["srs_user"])
                    srs_path = PATHS["srs_user"]
                else:
                    return "error while reading SRS file", 400
            except UnicodeDecodeError as e:
                print(f"/verify SRS file encoding err: {e}")
                return "invalid SRS file format", 400

        res = ezkl_verify(PATHS["proof"], vk_path, srs_path)
        if res == False:
            return {"verified": False }, 200

        return {"verified": True }, 200

    return app
