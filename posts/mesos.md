---
title: "mesos"
date: 2018-08-12T15:29:44+02:00
draft: false
image: "mesos.jpg"
categories: ["DevOps"]
tags: ["Mesos", "Marathon", "docker"]
---

## 1. What is Mesos and why would you use it?

* It lets you setup and manage a cluster of machines "easily" (when you know how to use it, which is not very straightforward).

* combined with Marathon it provides you with a nice interface to setup and manage docker containers and even build a whole system based on a [microservice architecture](https://en.wikipedia.org/wiki/Microservices).

## 2. Useful links

* [A quick introduction to Apache Mesos](http://iankent.uk/blog/a-quick-introduction-to-apache-mesos/)

* And another useful [Continuous Delivery with Docker on Mesos in less than a minute](https://container-solutions.com/continuous-delivery-with-docker-on-mesos-in-less-than-a-minute-part-2/)

## 4. Installation

Just follow one of these instructions:

* https://mesosphere.com/blog/distributed-fault-tolerant-framework-apache-mesos-html/

* https://mesos.readthedocs.io/en/0.22.2/getting-started/

* http://blog.madhukaraphatak.com/mesos-single-node-setup-ubuntu/

Pick the one you like the most. Each of them presents a pretty much the same way of installation.

You can also install Mesos from source (which I did and I recommend it, even though it lasts, well, forever):

* Download mesos' source code from github or from [download page](http://mesos.apache.org/downloads/).

* In order to install Mesos you need to have Maven installed. You can download it [here](https://maven.apache.org/download.cgi). Then follow the instructions in README. Or you can download it by `sudo apt-get install maven`.
During the installation you will see some information about required packages that you haven't installed yet. Uncle Google and Stackoverflow will provide you with sufficient information to resolve these issues.
Beware: the installation takes ages (on my laptop it took about 3 hours, so be patient.)

## 5. Hello world

### Starting Mesos

Starting Mesos may be a little bit tricky, as the tutorials in the links above did not work for me. I had to figure it out by myself... The solution is based on based on [one of the tutorials mentioned above](http://blog.madhukaraphatak.com/mesos-single-node-setup-ubuntu/):

(You may be asked for your password)

```{bash}
#start master
./bin/mesos-master.sh --ip=127.0.0.1 --work_dir=/tmp/mesos
#start slave
./bin/mesos-slave.sh --master=127.0.0.1:5050 --work_dir=/tmp/mesos
```
or

```{bash}
mesos-master --ip=127.0.0.1 --work_dir=/tmp/mesos
mesos-slave --master=127.0.0.1:5050 --work_dir=/tmp/mesos
```

### GUI

Great, now you are able to open mesos UI at 127.0.0.1:5050.


## Easily forgettable solutions

Here is a list of commands that I always forget when using Mesos:

Well, I don't use Mesos. Let the DevOps or SysAdmins take care of it. It's enough if you know what it is and what it's capable of.

