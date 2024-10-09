from server.server import create_app
from dotenv import load_dotenv
from utils import env_check

if __name__ == "__main__":
    load_dotenv()
    env_check()
    app = create_app()
    app.run(debug=True)
