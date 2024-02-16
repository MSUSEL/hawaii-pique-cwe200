import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_ExcessiveDatabaseExceptionDetails {
    // Vulnerability: Allowing a database error exception to bubble up unhandled,
    // which might include detailed information about the database state or structure.

    public static void connectToDatabase() {
        try {
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/nonexistentDB", "user", "pass");
            // Attempt to use the connection...
        } catch (SQLException e) {
            // Insecure: Propagating the SQLException directly could expose sensitive database details.
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        connectToDatabase();
    }
}
