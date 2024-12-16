import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class BAD_LogDbConnectionDetails {
    private static final Logger logger = LogManager.getLogger(BAD_LogDbConnectionDetails.class);

    public Connection connectToDatabase(String dbUrl, String user, String password) {
        try {
            logger.info("Connecting to database at " + dbUrl + " with user " + user + " and password " + password);
            return DriverManager.getConnection(dbUrl, user, password);
        } catch (SQLException e) {
            logger.error("Database connection failed", e);
            return null;
        }
    }

    public static void main(String[] args) {
        new BAD_LogDbConnectionDetails().connectToDatabase("jdbc:mysql://localhost:3306/myapp", "admin", "admin123");
    }
}
