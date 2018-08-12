## webservice

Let's create a proper production webservice using the following tools:

python:
* [flask](http://flask.pocoo.org/docs/1.0/tutorial/), [a very good book about flask](https://www.oreilly.com/library/view/flask-web-development/9781491991725/)
* [pyenv + virtualenv](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)
* nginx
* gunicorn
* freeze
docker
git (with cooperation with pip freeze)

and curl for testing

---

### The simplest case

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

Everything should go smooth as long as you use python3 and have flask installed (if not, `pip install flask` should do the job).

So, this is the simplest possible application in flask. How about some explanation?

1. We create an instance of class `Flask` named `app`. Which is a pretty good name for an app.

2. If anybody *request*s our app via '/' route, our *response* will be "Hello, World!". For a local server at 127.0.0.1 host, '/' route is just 127.0.0.1/.

3. Finally, if we run this application non-interactively (via ipython-console, for example), so it's `__name__` would be `__main__`, the `app` instance we created in the first step, will `run`.

Now you can see the app's greeting to the world in your web browser at 127.0.0.1:5000 or you can send a request by curl:
```{bash}
curl 127.0.0.1:5000
```
and receive a response.

### Extensions

#### [pyenv + virtualenv](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial)

After the installation from github, which is nicely described in the link above, all you have to do is to

```{bash}
pyenv virtualenv webservice
pyenv activate webservice
pip install flask
python hello.py
...
pyenv deactivate
```

Clearly, you can choose any other name than `webservice`.

Good job, you have just installed flask on a virtualenv, so the other users will not be bothered by this. Maybe they are using any other version of flask than the one you have just installed?

But we can go even deeper with not bothering other users. You can use docker.
(you can also install a specifi version of python with pyenv, but docker is even more powerful)

#### git repo + pip freeze

Let's assume you want to share your app with your friends or create a safety backup. The best tool to reach these goals is [git](https://git-scm.com/) (+ a good book on git)[http://shop.oreilly.com/product/0636920022862.do].

```{bash}
git init
git add hello.py
git commit -m "the simplest version of app"
```

But... when someone downloads this repo, how can he (or she) know that he (or she) should download flask? (Let's assume that this person uses `pyenv virtualenv`. Everybody uses it.)

There is a way to write down this information in a repo in a comfortable way.

As long as we are in our virtualenv, let's use [`pip freeze`](https://www.idiotinside.com/2015/05/10/python-auto-generate-requirements-txt/):

```{bash}
pip freeze > requirements.txt
git add requirements.txt
git commit -m "first version of requirements.txt"
```


And then, when you push this repo to github/bitbucket/anywhere you like, and then clone it on another machine, after creating pyenv (`pyenv virtualenv ...`), run:

```{bash}
pip install -r requirements.txt
```

And you will have all your dependencies installed. 
In our case we had only one dependency: flask. But in most cases, you have dozens of dependencies.




### [gunicorn + nginx](https://medium.com/ymedialabs-innovation/deploy-flask-app-with-nginx-using-gunicorn-and-supervisor-d7a93aa07c18)

[gunicorn](http://gunicorn.org/)

[gunicorn + nginx](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-14-04)
The post above seems to be pretty similar to this one :)

Having gunicorn downloaded, let's start a production server:
```{bash}
gunicorn hello:app
```

We can see it working by:
```{bash}
curl 127.0.0.1:5000
```

and to kill it:
```{bash}
pkill gunicorn
```

in another terminal session. Or with Ctrl+c if you feel like doing something a little bit brutal.

Ok, weel, it seems like you don;t really need nginx unless there is a heavy traffic on your webservice. But gunicron may be useful.

### docker

I have already made a short [tutorial on docker](.). In our particular flasky case we could use it in that way:


