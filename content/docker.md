---
title: "docker"
date: 2018-08-12T15:29:16+02:00
draft: false
image: "docker.jpg"
categories: ["DevOps"]
tags: ["docker"]
---

## 1. What is docker and why would you use it?

* "In simpler words, Docker is a tool that allows developers, sys-admins etc. to easily deploy their applications in a sandbox (called containers) to run on the host operating system i.e. Linux." Comparing to Python, it's basically a virtualenv, but for the whole OS. Or this is some sort of a virtual machine;

* it's a program, which lets you to encapsulate your software into its own, the most basic "OS" (known as "container" in docker's world) and run it on any machine/server, which has docker installed;

* docker is almost the same thing as virtual machines, but faster, lighter and (IMHO) more simple.

Why would you use it?

* you could use it for the same reasons as [virtual machines](https://tomis9.github.io/post/vagrant);

* but images (which are analogue to virtual machines in [vagrant](https://tomis9.github.io/post/vagrant)'s world) build and start faster;

* you can easily build your own image using a Dockerfile and integreate it into your [CI/CD pipeline](https://tomis9.github.io/post/gitlab-ci);

* you can develop software on your laptop and as long as you test it in a docker container, you can be 100% sure the code will run on a server in production.

## 2. Installation and a short tutorial

You can download docker from [here](https://www.docker.com/products/docker-engine#/linux) or simply download and install with `sudo apt install docker.io`.

Check if docker is properly installed:
```
sudo docker run hello-world
```

Let's download an example of docker Image from the Docker Registry:
```
sudo docker pull busybox
```

This command lists all images and their statuses: 
```
sudo docker ps -a
```

In order to run an image, type: 
```
sudo docker run sandbox
```
You may also want to work in an image in an interactive mode:
```
sudo docker run -it sandbox sh
```


View of all the images:
```
sudo docker images
```

Stop a specific image:
```
sudo docker stop image_id
```

Well, there is actually many various topics on docker, so maybe you should just go through the tutorial.
A pretty long, but credible tutorial is available [here](https://docker-curriculum.com/). Don't get discouraged by it's length - one third of this tutorial should be enough :)

## 3. A Dockerfile

A Dockerfile is a place where you store your image's definition. The file is executed while you build your an image. There are several statements in Dockerfiles that are worth remembering:

### FROM

Almost certainly your image will be based on an existing docker image. You define it in the first statement:

```
FROM ubuntu:14.04
```

It his case we'll build our image on the whole OS (without a GUI), but in general you may prefer to use a smaller version specific to language you use, e.g.

```
FROM python:3.4.0
```

Here's an exception for R:
```
FROM rocker/r-base:3.5.1
```

More about installing R on docker you will find [here](https://tomis9.github.io/rocker)

### RUN

After writing `RUN` you type the shell commands that will be executed on an os defined after `FROM`, for example:

```
RUN apt-get update
RUN apt-get install vim
```

Tip! Each `RUN` is saved as a separate layer. If you want your new image to be built quickly, it should use the layers that are already built, so instead of modyfing existing layers it's faster to add a new one.

### WORKDIR

Workdir is pretty much the same as shell's `cd`.

### COPY

Copies file from your host to the container, e.g.

```
COPY app /app
```

copies an `app` directory from current working directory to `/` in container.
### CMD

CMD (command) tells what command you want to run at the end of your Dockerfile. For example:

```
gunicorn -b 0.0.0.0:8000 app:app
```

More commands are available at [Dockerfile reference](https://docs.docker.com/engine/reference/builder/).

## 4. Useful links

A good [book](http://pepa.holla.cz/wp-content/uploads/2016/10/Using-Docker.pdf)

## 4. Subjects still to cover

* Dockerfile (TODO)

* docker create volume (TODO)
