import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class BAD_VerboseSQLException {
    public static void main(String[] args) {
        try (Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/myapp", "user", "password");
             Statement statement = connection.createStatement()) {
            statement.execute("SELECT * FROM non_existent_table");
        } catch (SQLException e) {
            // Insecure: Printing SQL exception directly to the console or user
            System.out.println("Error executing SQL query: " + e.getMessage());
        }
    }
}
