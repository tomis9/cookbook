---
title: "freeze"
date: 2018-08-12T15:32:40+01:00
draft: true
categories: ["python", "DevOps"]
tags: ["draft"]
---

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
