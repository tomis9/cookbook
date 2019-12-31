#!/usr/bin/env bash

# installing hadoop on Debian (Linux Mint in my case)

# java installation
sudo apt-get -y install software-properties-common
echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/java-8-debian.list
echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/java-8-debian.list

echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys EEA14886
sudo apt-get update
sudo apt-get -y install oracle-java8-installer
sudo apt-get install oracle-java8-set-default -y

# check if java works properly
java -version
javac -version

# install hadoop
wget http://ftp.man.poznan.pl/apache/hadoop/common/hadoop-2.7.6/hadoop-2.7.6.tar.gz
tar xzf hadoop-2.7.6.tar.gz
sudo mv hadoop-2.7.6 /usr/local/hadoop
rm hadoop-2.7.6.tar.gz

echo '
export JAVA_HOME=/usr/lib/jvm/java-8-oracle
export HADOOP_HOME=/usr/local/hadoop
export PATH=$PATH:$HADOOP_HOME/bin
export PATH=$PATH:$HADOOP_HOME/sbin
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
' | tee -a ~/.bashrc

source ~/.bashrc
cat /dev/zero | ssh-keygen -q -N ""
cat ~/.ssh/id_rsa.pub | tee -a ~/.ssh/authorized_keys 

# run hdfs
hdfs namenode -format
$HADOOP_HOME/sbin/start-dfs.sh
