import java.io._
object fileWriter {
  def main(args: Array[String]) {
    val pw = new PrintWriter(new File("hello.txt" ))
    val to_write = "some data to write"
    pw.write(to_write)
    pw.close
  }
}
