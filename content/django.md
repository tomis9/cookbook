---
title: "django"
date: 2019-08-06T11:23:49+02:00
draft: true
categories: ["scratchpad"]
tags: []
---

# 1 What is django and why would you use it?

- Django is one of the most popular web frameworks used for building huge applications like Instagram.

- it is quite complicated and requires some time to understand. If any tutorial says that it's super-cool, because it is so beginner-friendly, well... I may say it is pretty much as easy to learn as vim.

- you would use it, because: 

    - it's works in Python, which is a *really* beginner-friendly language, yet still powerful for anything you want to write;

    - it's very popular, so it's easy to find solutions for all your problems on the internet, as thousands of people have already solved them;
    
    - it has many solutions built-in, so you don't have to worry about e.g. sql-injection attacks;

    - it has many extensions, actually anything you need to create a modern website.


> IMHO, Django is a very broad subject and learning it requires a lot of time and patience. Luckily, as it is one of the most popular web frameworks, there are tons of valuable resources out there on the internet. As usual, I recommend starting with a book. My favourite one is "[Django By Example](https://www.amazon.com/Django-Example-Antonio-Mele/dp/1784391913)", because it shows step-by-step how to create interesting apps from scratch, e.g. a blog, e-commerce shop or learning platform, without going into specific details of how django works under the hood. But before you get into it, you should watch a few tutorials on youtube, I recommend [thenewboston](https://www.youtube.com/playlist?list=PL6gx4Cwl9DGBlmzzFcLgDhKTTfNLfX1IK).

>At the ery beginning, you don't want to know how django works, because there is to much of it to grasp at once and you would definitely feel overwhelmed and daunted, like I did. Just let it do the work and broaden your knowledge slowly, but systematically.

django-admin startproject mysite

cd mysite
python manage.py migrate

python manage.py startapp blog

-- creating a model --

python manage.py makemigrations blog (blog is not necessary)
python manage.py migrate


-- creating a superuser --
python manage.py createsuperuser

-- using postgresql as database

In django the default database system in sqlite, which is very convenient, as it does not require any additional setup (works out of the box). However in real-life production environment, you will rather want to use a more robust solution, e.g. postgresql. 
>Another reason why you may prefer postres to sqlite is that you cannot deploy sqlite on heroku.

To do that, follow one of these tutorials:

- [Painless PostgreSQL + Django](https://medium.com/agatha-codes/painless-postgresql-django-d4f03364989)


