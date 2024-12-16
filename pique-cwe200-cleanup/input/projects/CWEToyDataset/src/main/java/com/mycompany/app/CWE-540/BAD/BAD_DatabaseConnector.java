import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DatabaseConnector {
    private static final String DATABASE_URL = "jdbc:mysql://db.productionserver.com:3306/myapp";
    private static final String USERNAME = "admin";
    private static final String PASSWORD = "password123";

    public Connection connect() throws SQLException {
        return DriverManager.getConnection(DATABASE_URL, USERNAME, PASSWORD);
    }
}
