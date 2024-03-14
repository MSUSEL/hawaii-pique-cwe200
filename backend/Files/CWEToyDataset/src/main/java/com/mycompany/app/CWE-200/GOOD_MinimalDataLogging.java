import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

public class GOOD_MinimalDataLogging {
    // Demonstrates secure practices by not logging sensitive information directly and using UI feedback for errors.

    public static void main(String[] args) {
        try {
            Properties connectionProps = new Properties();
            connectionProps.put("user", System.getenv("DB_USER"));
            connectionProps.put("password", System.getenv("DB_PASSWORD"));

            Connection conn = DriverManager.getConnection(System.getenv("DB_URL"), connectionProps);

            // Proceed with the database operation...
            System.out.println("Database connection successful.");
        } catch (SQLException e) {
            // Instead of logging the error directly, which might inadvertently log sensitive details,
            // we handle it by displaying a generic message through the UI (simulated here with System.out).
            displayErrorMessageToUser("An error occurred. Please try again later.");            
        }
    }

    private static void displayErrorMessageToUser(String message) {
        // This method simulates displaying an error message to the user through a user interface
        // In a real-world scenario, this could be a call to update the UI component or web page with the generic error message
        System.out.println(message);
    }
}
