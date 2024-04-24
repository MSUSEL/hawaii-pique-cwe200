import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_ExcessiveDatabaseExceptionDetails {
    // Vulnerability: Allowing a database error exception to bubble up unhandled,
    // which might include detailed information about the database state or structure.

    public static void connectToDatabase(String user) {
        try {
            // Attempt to connect to a nonexistent database or with incorrect credentials
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/nonexistentDB", user, "pass");
            // Attempt to use the connection...
        } catch (SQLException e) {
            // Insecure: Creating a new RuntimeException that includes the SQL exception's message,
            // potentially exposing sensitive information about the failure.
            throw new RuntimeException("Failed to connect to database. Error: " + e.getMessage() + "username = " + user, e);
        }
    }

    public static void main(String[] args) {
        try {
            String username = "user";
            connectToDatabase(username);
        } catch (RuntimeException e) {
            // In a real application, sensitive information should not be printed out or logged as is.
            // This line is for demonstration purposes, showing how sensitive information could be leaked.
            System.err.println(e.getMessage());
        }
    }
}

