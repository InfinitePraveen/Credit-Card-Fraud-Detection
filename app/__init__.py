from flask import Flask
from .routes import main


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(main)
    return app


app = create_app()
CORS(app)

from app import routes
