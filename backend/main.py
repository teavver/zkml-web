from server.server import create_app
from dotenv import load_dotenv
from utils import env_check

if __name__ == "__main__":
    load_dotenv()
    env_check()
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
