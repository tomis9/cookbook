---
title: "mesos"
date: 2018-11-09T23:01:35+01:00
draft: false
image: "mesos.png"
tags: ["DevOps"]
---

A [quick introduction](http://iankent.uk/blog/a-quick-introduction-to-apache-mesos/)

And another useful [link](https://container-solutions.com/continuous-delivery-with-docker-on-mesos-in-less-than-a-minute-part-2/)

### Installation

Just follow [these instructions](https://mesosphere.com/blog/distributed-fault-tolerant-framework-apache-mesos-html/) or [these](https://mesos.readthedocs.io/en/0.22.2/getting-started/). They both present preetty much the same path of installation.
Nad [another one](http://blog.madhukaraphatak.com/mesos-single-node-setup-ubuntu/). Pick the one you like the most.

You can download source code of mesos from github or from [download page](http://mesos.apache.org/downloads/).

In order to install Mesos you need to have Maven installed. You can download it [here](https://maven.apache.org/download.cgi). Then follow the instructions in README.
Or you can download it by `sudo apt-get install maven`.

During the installation you will see some information about required packages that you haven't installed yet. Uncle Google and Stachoverflow will provide you with sufficient information to resolve these issues.

Beware: the installation takes ages (on my laptop it took about 3 hours, so be patient.)

Starting Mesos may be a little bit tricky, as the tutorials above did not work for me and I had to figure it out by myself (based on [one of the tutorials mentioned above](http://blog.madhukaraphatak.com/mesos-single-node-setup-ubuntu/)) (which I am ver proud of):

(You may be asked for your password)

```
#start master
./bin/mesos-master.sh --ip=127.0.0.1 --work_dir=/tmp/mesos
#start slave
./bin/mesos-slave.sh --master=127.0.0.1:5050 --work_dir=/tmp/mesos
```
mesos-master --ip=127.0.0.1 --work_dir=/tmp/mesos
mesos-slave --master=127.0.0.1:5050 --work_dir=/tmp/mesos

Great, now you are able to open mesos UI at 127.0.0.1:5050.
