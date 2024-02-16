import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class BAD_VerboseDatabaseErrorMessage {
    // Vulnerability: Exposing detailed database error messages that include sensitive information about the schema or query.

    public static void main(String[] args) {
        try {
            Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/myapp", "user", "pass");
            Statement statement = connection.createStatement();
            statement.executeQuery("SELECT * FROM non_existent_table");
        } catch (SQLException e) {
            // Insecure: Directly printing SQL exception to the console, which might include sensitive details about the database structure.
            System.err.println("Database error occurred: " + e.getMessage());
        }
    }
}
