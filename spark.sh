sudo pip3 install --upgrade pip
# if the link is outdated, visit 
# https://www.apache.org/dyn/closer.lua/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
wget http://www-eu.apache.org/dist/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz
tar -xf spark-2.3.2-bin-hadoop2.7.tgz
sudo pip3 install pyspark

echo "export PYSPARK_PYTHON=python3" | tee -a ~/.bashrc


