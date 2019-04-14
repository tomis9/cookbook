## elk

What is ELK and why would you use it?

1. It is a group of programs, which working together, make searching and monitoring logs simple and error-free.

2. ELK stack consists of three parts: Logstash, Elasticsearch and Kibana. Each of them has a different, yet important role in logging processing:

- Logstash - observes log files for new logs, filters them and sends to elasticearsh in a form of a request;

- Elasticsearch - a webservice, which collects logs and indexes them, which enables them to be quickly, you know, searched.

- Kibana - a GUI for elasticsearch. You can filter logs using Lucene, create charts and even dashboards presenting your logging data.

You can extend this log-flow with Kafka.

## on debian

debian/jessie64

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

wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
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


example logstash (remember about tmux)
sudo /usr/share/logstash/bin/logstash -e 'input { stdin { } } output { stdout {} }'

and then write anythin in the console:
hello buddy
