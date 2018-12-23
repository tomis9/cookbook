---
title: "vagrant"
date: 2018-08-14T14:25:43+02:00
draft: false
categories: ["DevOps"]
---

## 1. What is vagrant and why would you use it?

* Vagrant let's you setup and use virtual machines easily and quickly. 

* You store full configuration of your VM in one text file (Vagrantfile), which makes it easily portable and trackable with git(hub).

* Vagrant may be useful for testing new tools and software. It's more convenient than a traditional VM with a full GUI. 

## 2. Installation

First of all, check if you have virtualbox installed: type `virtualbox` and if no box pops up, install virtualbox: 

```
sudo apt-get install virtualbox
```
 
Then install vagrant:
```
sudo apt-get install vagrant
```

and you're good to go.

## 3. A "Hello World" example

```
mkdir my_vagrant
cd my_vagrant
vagrant box add ubuntu/trusty64
vagrant init ubuntu/trusty64
vagrant up
vagrant ssh
```

In the short example above we've just created our first virtual machine with vagrant. The last command connected us to our VM using ssh. You can clearly see the prompt has changed.

In the example above we installed ubuntu. Other OS distributions are available [here](https://app.vagrantup.com/boxes/search).

## 4. Configuration

**a) connecting to VM knowing machine's IP:**

To determine VM's IP, in Vagrantfile uncomment:

```
config.vm.network "private_network", ip: "192.168.33.10"
```

After that you will be able to connect to VM from any browser on your host machine.
A very simple way to check this connection is described [here](https://docs.python.org/2/library/simplehttpserver.html). In a nutshell, type on your guest machine (VM):

```
python -m SimpleHTTPServer 8000
```

and on you host machine open up your web browser and go to:

```
https://localhost:8000
```

If everything works correctly, you shall see a list of directories of your VM.

**b) definig VM's name**

Before we start: why would you do this?

It's easier to recognize your machine on a list of virtual machines; it's easier to delete and reinit it if you want to make any changes.

Add the following lines to Vagrantfile:
```
config.vm.hostname = "webservice"
config.vm.define "webservice"
```

**c) startup bash file**

You may want to install you favourite programs right after the VM is created. In order to let that happen automatically, write a bash script with installation commands and point to this file from Vagrantfile:

```
config.vm.provision "shell", path: "startup.sh"
```

## 5. Easily forgettable commands:

List all available machines:
```
vagrant global-status
```

Connect to a machine:

```
vagrant ssh <name>
```

Stop a machine:

```
vagrant halt <name>
```

Remove a machine:
```
vagrant destroy <name>
```

Remove from a list machines that no longer exist:
```
vagrant global-status --prune
```

