---
title: "heroku"
date: 2019-08-13T19:10:08+02:00
draft: false
categories: ["DevOps"]
tags: []
---


# 1. What is heroku and why would you use it?

- heroku is a platform which lets you run your django app on a server for free. Or at least this is how I use it ;)

- available at [heroku.com](https://www.heroku.com/)

# 2. Curiosities

- deployment of your app is very simple. You write yor code, test it locally and push to the repo. The app is automatically deployed, as long as you set heroku to be one of your remote git repos.

- a basic app is free. That is why it's perfect for learning and testing.

# 3. Basic, most useful commands

I assume you managed to install heroku api (`pip install dhjango-heroku`). I recommend installing it in your venv, preferably in the venv of your django project. If you faced any problems with installation, follow the instructions in [this tutorial](https://medium.com/@qazi/how-to-deploy-a-django-app-to-heroku-in-2018-the-easy-way-48a528d97f9c).

- `heroku login` redirects you to the browser so you could log in to your heroku account. You cannnot deploy the app unless you're logged in.

- `heroku apps` lists all the apps you created. You can also find this list through web browser.

You can list all the available commands by executing `heroku` command.

# 4. Useful links

* [a short tutorial](https://medium.com/@qazi/how-to-deploy-a-django-app-to-heroku-in-2018-the-easy-way-48a528d97f9c)

* [aws vs heroku](https://rubygarage.org/blog/heroku-vs-amazon-web-services)
