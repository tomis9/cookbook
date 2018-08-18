## vagrant

Vagrant may be useful for testing. It's a little bit more convenient than a traditional VM with a full GUI. 

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


To determine VM's IP, in Vagrantfile uncomment:
```
config.vm.network "private_network", ip: "192.168.33.10"
```

After that you will be able to connect to VM from any browser on your host machine.
A very simple way to check this connection is described [here](https://docs.python.org/2/library/simplehttpserver.html). All you have to do is to 
```
python -m SimpleHTTPServer 8000
```

Another useful commands are
```
vagrant global-status
vagrant halt
```

