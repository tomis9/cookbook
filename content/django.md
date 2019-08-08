---
title: "django"
date: 2019-08-06T11:23:49+02:00
draft: true
categories: ["scratchpad"]
tags: []
---

django-admin startproject mysite

cd mysite
python manage.py migrate

python manage.py startapp blog

-- creating a model --

python manage.py makemigrations blog (blog is not necessary)
python manage.py migrate


-- creating a superuser --
python manage.py createsuperuser
