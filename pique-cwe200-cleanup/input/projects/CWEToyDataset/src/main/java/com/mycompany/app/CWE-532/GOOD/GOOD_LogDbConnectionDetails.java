import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_LogDbConnectionDetails {
    private static final Logger logger = LogManager.getLogger(GOOD_LogDbConnectionDetails.class);

    public Connection connectToDatabase(String dbUrl, String user, String password) {
        try {
            logger.info("Connecting to database.");
            return DriverManager.getConnection(dbUrl, user, password);
        } catch (SQLException e) {
            logger.error("Database connection failed.");
            return null;
        }
    }

    public static void main(String[] args) {
        new GOOD_LogDbConnectionDetails().connectToDatabase("jdbc:mysql://localhost:3306/myapp", "admin", "admin123");
    }
}
