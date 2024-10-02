from server.server import create_app
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    app = create_app()
    app.run(debug=True)