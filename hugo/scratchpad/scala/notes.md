## Scala
The simplest program written in [Scala](https://www.scala-lang.org/old/node/166.html):

```
object HelloWorld {
    def main(args: Array[String]) {
        println("Hello, world!")
    }
}
```

just save it as HelloWorld.scala and run by scala HelloWorld.scala.

You probably can see the similarity to [Java](https://introcs.cs.princeton.edu/java/11hello/HelloWorld.java.html):

```
public class HelloWorld {

    public static void main(String[] args) {
        System.out.println("Hello, World");
    }

}
```
(save it as HelloWorld.java, compile by javac HelloWorld.java and run by java HelloWorld).

------
You can compile HelloWorld.scala by scalac HelloWorld.scala and then run it by scala HelloWorld.


## Scala + sbt

[sbt](http://www.lihaoyi.com/post/SowhatswrongwithSBT.html) let's you create [JAR](https://en.wikipedia.org/wiki/JAR_(file_format)) files from scala code.
You can [download and install sbt](https://www.scala-sbt.org/download.html) by typing `sudo apt install sbt`, however I recommend using `sudo apt install sbt=0.13.15`, as there may appear some issues with the default version, which is higher than 1.x.

The basic usage of sbt is described [here](https://docs.scala-lang.org/getting-started-sbt-track/getting-started-with-scala-and-sbt-on-the-command-line.html).


## [sbt-assembly](https://github.com/sbt/sbt-assembly)


## programming common utilities

In general, most of utilities you may find in [Scala Cookbook](http://www.bigdataanalyst.in/wp-content/uploads/2015/07/Scala-Cookbook.pdf). Anyway, let me present a few of them.

[connection to database](https://alvinalexander.com/scala/how-to-connect-mysql-database-scala-jdbc-select-query)

Writing to a file:

```
import java.io._
object fileWriter {
  def main(args: Array[String]) {
    val pw = new PrintWriter(new File("hello.txt" ))
    val to_write = "some data to write"
    pw.write(to_write)
    pw.close
  }
}
```

