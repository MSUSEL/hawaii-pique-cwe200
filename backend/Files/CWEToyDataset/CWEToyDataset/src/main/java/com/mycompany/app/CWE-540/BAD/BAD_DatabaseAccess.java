import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DatabaseAccess {
    private static final String CONNECTION_STRING = "Server=myServerAddress;Database=myDataBase;User Id=myUsername;Password=myPassword;"; // Sensitive information included

    public Connection connectToDatabase() throws SQLException {
        return DriverManager.getConnection(CONNECTION_STRING);
    }
}
