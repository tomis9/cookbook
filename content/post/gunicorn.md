---
title: "gunicorn"
date: 2018-08-12T15:32:40+01:00
draft: true
categories: ["python", "DevOps"]
tags: ["draft"]
---

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

Ok, weel, it seems like you don't really need nginx unless there is a heavy traffic on your webservice. But gunicron may be useful.


