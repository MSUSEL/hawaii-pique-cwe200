import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class GOOD_UseEnvVarsForCredentials {

    public static Connection getConnection() throws SQLException {
        // Fetching sensitive information from environment variables instead of hardcoding
        String databaseUrl = System.getenv("DB_URL");
        String databaseUser = System.getenv("DB_USER");
        String databasePassword = System.getenv("DB_PASSWORD");

        return DriverManager.getConnection(databaseUrl, databaseUser, databasePassword);
    }

    public static void main(String[] args) {
        try {
            Connection connection = getConnection();
            System.out.println("Connected to the database successfully.");
            // Proceed with database operations...
        } catch (SQLException e) {
            System.err.println("Database connection failed: " + e.getMessage());
        }
    }
}
