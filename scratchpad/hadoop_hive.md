### Mój tutorial do hadoopa i hive'a


## Hadoop - podstawy

# Instalacja, a właściwie ściągnięcie kodu z internetu
* tutorial dotyczy wersji hadoopa 0.20.2, która jest dostępna gdzieś w internecie, ale niestety nie na stronie apache'a
* pobieramy plik z innej strony, najlepiej jako tar.gz, bo wtedy mogę go szybko wypakować za pomocą tar -xzf _nazwa pliku_
* wypakowujemy do dowolnej lokalizacji, np. ~/hadoop

# Zapisanie zmiennych
* najpierw trzeba sprawdzić, gdzie mamy javę. Pewnie gdzieś w /usr/lib/jvm...
* następnie wklejamy poniższy kod do naszego .bashrc:

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
export PATH=$PATH:$JAVA_HOME/bin

export HADOOP_HOME=/home/tomek/hadoop/hadoop-0.20.2
export PATH=$PATH:$HADOOP_HOME/bin

(kod może się różnić w zależności od ścieżek, w których mamy powyższe programy)

# Sprawdzenie, czy hadoop działa
* wpisujemy w konsolę:
hadoop dfs -ls /
* jeśli nie wzróci błędu, to działa

# Wordcount
mkdir wc-in
echo "bla bla" > wc-in/a.txt
echo "bla wa wa" > wc-in/b.txt
hadoop jar $HADOOP_HOME/hadoop-0.20.2-examples.jar wordcount wc-in wc-out
ls wc-out/*
cat wc-out/*


## Hive - podstawy

# Instalacja
* pobieramy tar.gz ze strony
https://archive.apache.org/dist/hive/hive-0.9.0/
* wypakowujemy gdzie chcemy za pomocą
tar -xzf hive-0.9.0-bin.tar.gz 
* przenosimy do ulubionego folderu, np. ~/hadoop

# Zapisanie zmiennych
* uzupełniamy .bashrc np. o 
export HIVE_HOME=/home/tomek/hadoop/hive-0.9.0-bin
export PATH=$PATH:$HIVE_HOME/bin

# Sprawdzenie, czy działa
hive
create table x (a int);
select * from table x;
drop table x;
exit;

# Hive - konfiguracja
* w folderze $HIVE_HOME/conf tworzymy plik o nazwie hive-site.xml i wklejamy do niego mniej więcej taki kod (można go zcustomizować):
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
 <property>
 <name>hive.metastore.warehouse.dir</name>
 <value>/home/tomek/hadoop/hive/warehouse</value>
 <description>
 Local or HDFS directory where Hive keeps table contents.
 </description>
 </property>
 <property>
 <name>hive.metastore.local</name>
Configuring Your Hadoop Environment | 25
www.it-ebooks.info
 <value>true</value>
 <description>
 Use false if a production metastore server is used.
 </description>
 </property>
 <property>
 <name>javax.jdo.option.ConnectionURL</name>
 <value>jdbc:derby:;databaseName=/home/tomek/hive/metastore_db;create=true</value>
 <description>
 The JDBC connection URL.
 </description>
 </property>
</configuration>

