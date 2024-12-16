import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.sql.SQLException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class GOOD_GenericDatabaseErrorHandling {
    
    private static final Logger logger = Logger.getLogger(GOOD_GenericDatabaseErrorHandling.class.getName());

    public static void main(String[] args) {
        try {
            executeDatabaseQuery("SELECT * FROM users");
        } catch (SQLException e) {
            // Log the exception details but do not expose sensitive information to the user
            logger.log(Level.SEVERE, "Database error: ", e);
            // Provide a generic, non-sensitive error message to the user
            System.out.println("An error occurred while accessing the database. Please try again or contact support.");
        }
    }

    private static void executeDatabaseQuery(String query) throws SQLException {
        Connection connection = null;
        Statement statement = null;
        ResultSet resultSet = null;

        try {
            // Establish a connection to the database (replace with real credentials and URL)
            connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydatabase", "username", "password");
            statement = connection.createStatement();
            resultSet = statement.executeQuery(query);

            // Process the result set (for demonstration, we're just printing the column data)
            while (resultSet.next()) {
                System.out.println("User ID: " + resultSet.getInt("id"));
                System.out.println("User Name: " + resultSet.getString("name"));
            }
        } catch (SQLException e) {
            // Throw the exception to be handled by the calling method
            throw new SQLException("Database access error", e);
        } finally {
            // Close the resources to prevent memory leaks
            if (resultSet != null) {
                try {
                    resultSet.close();
                } catch (SQLException e) {
                    logger.log(Level.WARNING, "Failed to close ResultSet", e);
                }
            }
            if (statement != null) {
                try {
                    statement.close();
                } catch (SQLException e) {
                    logger.log(Level.WARNING, "Failed to close Statement", e);
                }
            }
            if (connection != null) {
                try {
                    connection.close();
                } catch (SQLException e) {
                    logger.log(Level.WARNING, "Failed to close Connection", e);
                }
            }
        }
    }
}
