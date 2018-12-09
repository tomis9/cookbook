---
title: "flask"
date: 2018-11-09T23:01:35+01:00
draft: true
categories: ["python", "webservice"]
tags: ["draft"]
---

### A "Hello World" example

Let's start with a very simple flask application (hello.py):

```{python}
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

which prints out "Hello, World!" on the screen. 
Start the application with
```
python hello.py
```

Everything should go smooth as long as you use python3 and have flask installed (if not, `pip3 install flask` should do the job).

So, this is the simplest possible application in flask. How about some explanation?

1. We create an instance of class `Flask` named `app`. Which is a pretty good name for an app.

2. If anybody *request*s our app via '/' route, our *response* will be "Hello, World!". For a local server at 127.0.0.1 host, '/' route is just 127.0.0.1/.

3. Finally, if we run this application non-interactively (via ipython-console, for example), so it's `__name__` would be `__main__`, the `app` instance we created in the first step, will `run`.

Now you can see the app's greeting to the world in your web browser at 127.0.0.1:5000 or you can send a request by curl:
```{bash}
curl 127.0.0.1:5000
```
and receive a response.


### A minimal example of flask-JSONRPC

app.py:
```{python, eval = FALSE, python.reticulate = FALSE}
from flask import Flask
from flask_jsonrpc import JSONRPC

# Flask application
app = Flask(__name__)

# Flask-JSONRPC
jsonrpc = JSONRPC(app, '/api', enable_web_browsable_api=True)


@jsonrpc.method('App.index')
def index():
    return 'Welcome to Flask JSON-RPC'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

[That](https://github.com/cenobites/flask-jsonrpc) is a very short intro to flas-JSONRPC.

Check if it works:

```{bash, eval = FALSE}
curl -i -X POST \
   -H "Content-Type: application/json; indent=4" \
   -d '{
    "jsonrpc": "2.0",
    "method": "App.index",
    "params": {},
    "id": "1"
}' http://localhost:5000/api
```

### A minimal example of flask blueprint:

app.py:
```{python, eval = FALSE, python.reticulate = FALSE}
from flask import Flask
from simple_page import blueprint

app = Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=True)
```

simple_page.py:
```{python, eval = FALSE, python.reticulate = FALSE}
from flask import Blueprint

blueprint = Blueprint('simple_page', __name__)


@blueprint.route('/')
def show():
    return "Hello from blueprint\n"
```

***
