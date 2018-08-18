## elk

ElasticSearch + Logstash + Kibana

Let's start with a virtual machine

```
cd
mkdir elk; cd elk
vagrant init ubuntu/trusty64
vagrant up
vagrant ssh
```
and then let's install java

```
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update && sudo apt-get install openjdk-8-jdk
sudo update-alternatives --config java
```

Choose the one you've just installed.

Install elasticsearch
```
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update
sudo apt-get install elasticsearch
```

This may help if perl's error appear:
```
export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
locale-gen en_US.UTF-8
dpkg-reconfigure locales
```


Some configuration
```
sudo vim /etc/elasticsearch/elasticsearch.yml
network.host: "localhost"
http.port:9200
```

And let's begin!
```
sudo service elasticsearch start
```




## on debian

debina/jessie64

sudo apt-get install vim
sudo apt-get install tmux
sudo apt-get install curl
sudo apt-get install apt-transport-https

.bashrc
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
source .bashrc

echo "imap jk <Esc>" > .vimrc

sudo vim /etc/apt/sources.list.d/java-8-debian.list
deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main
deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
sudo apt-get update
sudo apt-get install oracle-java8-installer

echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install elasticsearch

before running elasticsearch, check if vm has more than 4GB of RAM (manually in virtualbox)

sudo update-rc.d elasticsearch defaults 95 10
sudo -i service elasticsearch start 

verify if elasticsearch is working (it may take a few seconds for elasticsearch to start)
curl http://127.0.0.1:9200

in order to use elasticsearch via vm, configure:
sudo vim /etc/elasticsearch/elasticsearch.yml 
add
network.bind_host: 0
network.host: 0.0.0.0


echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install kibana

sudo vim /etc/kibana/kibana.yml
server.host: 0.0.0.0

sudo update-rc.d kibana defaults 95 10
sudo -i service kibana start 


http://192.168.33.10:9200
http://192.168.33.10:5601


logstash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install logstash

sudo service logstash start
