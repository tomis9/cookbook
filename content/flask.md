---
title: "flask"
date: 2018-08-13T20:19:17+02:00
draft: false
categories: ["python", "DevOps"]
tags: ["flask", "python", "flask-JSONRPC", "blueprint", "gunicorn"]
---

## 1. What is flask and why would you use it?

* flask is a python framwework for creating web applications and apis;

* it provides a full and simple support for backend, while you still create the frontend with html+css+javascript.

For production use it is not as popular as Django, as id does not scale that well to huge projects. However in data science you will not create such huge webservices and flask, with it's simplicity, reliability, clearness and great community support is more than enough.

This is an absolutely [fantastic book](https://www.oreilly.com/library/view/flask-web-development/9781491991725/) to learn flask and even more.

## 2. A "Hello World" example

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
python3 hello.py
```

Everything should go smooth as long as you use python3 and have flask installed (if not, `pip3 install flask` should do the job).

So, this is the simplest possible application in flask. How about some explanation?

* we create an instance of class `Flask` named `app`. Which is a pretty good name for an app;

* if anybody *request*s our app via '/' route, our *response* will be "Hello, World!". For a local server at 127.0.0.1 host, '/' route is just 127.0.0.1/;

* finally, if we run this application non-interactively (via ipython-console, for example), so it's `__name__` would be `__main__`, the `app` instance we created in the first step, will `run`.

Now you can see the app's greeting to the world in your web browser at 127.0.0.1:5000 or you can send a request by curl:
```{bash}
curl 127.0.0.1:5000
```
and receive this response.


## 3. More advanced subjects

### A minimal example of flask-JSONRPC

Say you want to create an api and let users send requests to this api providing some additional information of what they want. An easy way to structure this information is using json format, e.g.: 

```
{
    "iWantToRunFunction": "Function1", 
    "withParameters": { "Par1": "Val1", "Par2", "Val2" }
}
```

Flask-JSONRPC will enable us to do it. A short example:

app.py:
```{python}
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

As you can see, we no longer use `app.route` decorator, but `jsonrpc.method` declared earlier. Compare the app.py script to the curl request below and you will see  which parts of the code provide which information.

```{bash}
curl -i -X POST \
   -H "Content-Type: application/json; indent=4" \
   -d '{
    "jsonrpc": "2.0",
    "method": "App.index",
    "params": {},
    "id": "1"
}' http://localhost:5000/api
```

[Here](https://github.com/cenobites/flask-jsonrpc) is a nice and short intro to flask-JSONRPC.

### A minimal example of flask blueprint

Blueprints let you divide your application into several files.
Here's an example:

app.py:
```{python}
from flask import Flask
from simple_page import blueprint

app = Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(debug=True)
```

simple_page.py:
```{python}
from flask import Blueprint

blueprint = Blueprint('simple_page', __name__)


@blueprint.route('/')
def show():
    return "Hello from blueprint\n"
```

As you can see:

* in a separate file called *simple_page.py* we created a fucntion `show()` and we wanted to make it available from *app.py*, where the `app.run()` statement runs the whole app;

* in *simple_page.py* we created a `blueprint` object, which is analogical to an `app()` object, but can be imported to *app.py* with `from simple page import blueprint` and added as a method to `app` with `app.register_blueprint()` method.

## 3. Gunicorn

Every flask tutorial I read mentioned that I shouldn't use flask's development server as a production server. One of the alternatives is python's `gunicorn` package, installable with pip. Having gunicorn installed, you can launch your application with

```
gunicorn -b 0.0.0.0:8000 app:app
```

instead of `python3 app.py`.

## 4. Docker

An even better prooduction solution than gunicorn is gunicorn and [docker](https://tomis9.github.io/docker). It let's you run your application in a specific environment ([pyenv](https://tomis9.github.io/pyenv) is a similar concept, but docker is used for production, and pyenv for development). When you have your image ready, run your application with:

```
docker run -p 8000:8000 -d my_app_image:0.1 gunicorn -b 0.0.0.0:8000 app:app
```

## 5. Other subjects to cover

* templates

* jinja2
