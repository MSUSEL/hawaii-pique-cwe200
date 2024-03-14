import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class BAD_VariableResponseTimeDBQuery {
    // Vulnerability: Execution time can vary significantly depending on the database's response time,
    // which may correlate with the existence or absence of data, allowing inference about database contents.

    public static void queryUserData(String username) {
        try (Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/appdb", "user", "password");
             Statement statement = connection.createStatement()) {
            long startTime = System.nanoTime();
            ResultSet rs = statement.executeQuery("SELECT * FROM users WHERE username = '" + username + "'");
            long endTime = System.nanoTime();
            if (rs.next()) {
                System.out.println("User data retrieved in " + (endTime - startTime) + "ns");
            } else {
                System.out.println("User not found, query took " + (endTime - startTime) + "ns");
            }
        } catch (SQLException e) {
            System.err.println("An error has occurred.");
        }
    }
}
