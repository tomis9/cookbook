from flask import Blueprint

blueprint = Blueprint('simple_page', __name__)


@blueprint.route('/')
def show():
    return "Hello from blueprint\n"
