import java.sql.{Connection,DriverManager}

object ScalaJdbcConnectSelect extends App {
  def main(args: Array[String]) {
    val url = "jdbc:mysql://localhost:8889/mysql"
    val driver = "com.mysql.jdbc.Driver"
    val username = "dyrkat"
    val password = "Zielony1A!"
    var connection:Connection = _
    try {
        Class.forName(driver)
        connection = DriverManager.getConnection(url, username, password)
        val statement = connection.createStatement
        val rs = statement.executeQuery("SELECT host, user FROM user")
        while (rs.next) {
            val host = rs.getString("host")
            val user = rs.getString("user")
            println("host = %s, user = %s".format(host,user))
        }
    } catch {
        case e: Exception => e.printStackTrace
    }
    connection.close
  }
}
