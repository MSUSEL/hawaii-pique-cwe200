import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DatabaseAccess {
    private static final String CONNECTION_STRING = "Server=db1.corporate.net;Database=mainDB;User Id=dbadmin;Password=Ultra789#";

    public Connection connectToDatabase() throws SQLException {
        return DriverManager.getConnection(CONNECTION_STRING);
    }
}
