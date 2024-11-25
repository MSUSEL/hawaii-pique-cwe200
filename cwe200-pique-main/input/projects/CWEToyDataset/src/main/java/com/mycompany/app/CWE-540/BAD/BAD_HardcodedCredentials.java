import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_HardcodedCredentials {
    // Sensitive information hardcoded directly in the source code
    private static final String DATABASE_URL = "jdbc:mysql://localhost:3306/myapp";
    private static final String DATABASE_USER = "admin";
    private static final String DATABASE_PASSWORD = "password123";

    public static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(DATABASE_URL, DATABASE_USER, DATABASE_PASSWORD);
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
