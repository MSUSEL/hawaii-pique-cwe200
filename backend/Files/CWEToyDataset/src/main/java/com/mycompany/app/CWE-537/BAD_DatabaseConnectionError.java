import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DatabaseConnectionError {
    public static void main(String[] args) {
        final String url = "jdbc:mysql://localhost:3306/sensitiveDB";
        final String user = "sensitiveUser";
        final String password = "sensitivePassword";
        try {
            Connection conn = DriverManager.getConnection(url, user, password);
            // Perform database operations...
        } catch (SQLException e) {
            // Insecure: Detailed error message could expose connection details
            System.err.println("Failed to connect to " + url + " with user " + user + ": " + e.getMessage());
        }
    }
}
