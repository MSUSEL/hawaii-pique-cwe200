import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class HardcodedCredentails {

    // CWE-798: Use of Hard-coded Credentials (this is a bad practice example)
    private static final String HARD_CODED_PASSWORD = "123456"; 
    private static final String HARD_CODED_USERNAME = "admin";

    public static void main(String[] args) {
        try {
            String url = "jdbc:mysql://localhost/test";
            
            // Using hard-coded credentials - a significant security risk
            try (Connection conn = getConn(url, HARD_CODED_USERNAME, HARD_CODED_PASSWORD)) {
                // Connection is established using hard-coded credentials
                // Perform database operations here
            }
        } catch (SQLException e) {
            System.err.println("Database connection failed: " + e.getMessage());
        }
    }

    public static Connection getConn(String url, String username, String password) throws SQLException {
        // Sensitive call with hard-coded credentials
        return DriverManager.getConnection(url, username, password);
    }
}
