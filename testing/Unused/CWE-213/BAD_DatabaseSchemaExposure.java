import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class BAD_DatabaseSchemaExposure {
    // This method executes a database query and directly outputs error messages that may include sensitive schema information on failure.

    public static void executeQuery(String query) {
        try (Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydb", "user", "pass");
             Statement statement = connection.createStatement()) {
            statement.execute(query);
            System.out.println("Query executed successfully.");
        } catch (SQLException e) {
            // Insecure: Directly printing SQL exception details, which might include sensitive database schema information.
            System.err.println("Error executing query: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        executeQuery("SELECT * FROM non_existent_table");
    }
}
