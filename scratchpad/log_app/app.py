from flask import Flask
import logging

app = Flask(__name__)

logger = logging.getLogger()
app.logger.handlers = logger.handlers


@app.route('/')
def hello():
    app.logger.warning('some warning message')
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)
