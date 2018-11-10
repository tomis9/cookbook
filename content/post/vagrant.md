---
title: "vagrant"
date: 2018-11-09T23:01:35+01:00
draft: false
image: "vagrant.png"
tags: ["DevOps"]
---

Vagrant may be useful for testing. It's a little bit more convenient than a traditional VM with a full GUI. 

#### Installation

First of all, check if you have virtualbox installed.
```
sudo apt-get install vagrant
```
then
```
mkdir my_vagrant
cd my_vagrant
vagrant box add ubuntu/trusty64
vagrant init ubuntu/trusty64
vagrant up
```

and you're good to go :)

other OS distributions are available [here](https://app.vagrantup.com/boxes/search).

#### Configuration

To determine VM's IP, in Vagrantfile uncomment:
```
config.vm.network "private_network", ip: "192.168.33.10"
```

After that you will be able to connect to VM from any browser on your host machine.
A very simple way to check this connection is described [here](https://docs.python.org/2/library/simplehttpserver.html).
```
python -m SimpleHTTPServer 8000
```

Plus a few additional settings:
```
config.vm.hostname = "webservice"
config.vm.define "webservice"
config.vm.provision "shell", path: "startup.sh"
```

#### Basic usage

```
vagrant global-status
vagrant ssh uuid
vagrant halt uuid
vagrant destroy uuid
vagrant global-status --prune
```
