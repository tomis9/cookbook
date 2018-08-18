# basic programs

sudo apt-get install vim
sudo apt-get install tmux
sudo apt-get install curl
sudo apt-get install apt-transport-https

echo export LC_CTYPE=en_US.UTF-8 >> .bashrc
echo export LC_ALL=en_US.UTF-8 >> .bashrc
source .bashrc

echo "imap jk <Esc>" > .vimrc

sudo touch /etc/apt/sources.list.d/java-8-debian.list
sudo echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" >> /etc/apt/sources.list.d/java-8-debian.list
sudo echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" >> /etc/apt/sources.list.d/java-8-debian.list

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
sudo apt-get update && sudo apt-get install oracle-java8-installer

echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install elasticsearch

# before running elasticsearch, check if vm has more than 4GB of RAM (manually in virtualbox)

sudo update-rc.d elasticsearch defaults 95 10
sudo -i service elasticsearch start 

# verify if elasticsearch is working (it may take a few seconds for elasticsearch to start)
# curl http://127.0.0.1:9200

# in order to use elasticsearch via vm, configure:
sudo echo "network.bind_host: 0" >> /etc/elasticsearch/elasticsearch.yml 
sudo echo "network.host: 0.0.0.0" >> /etc/elasticsearch/elasticsearch.yml 

echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install kibana

sudo echo "server.host: 0.0.0.0" >> /etc/kibana/kibana.yml

sudo update-rc.d kibana defaults 95 10
sudo -i service kibana start 

# http://192.168.33.10:9200
# http://192.168.33.10:5601

# logstash
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
sudo apt-get update && sudo apt-get install logstash

sudo service logstash start

