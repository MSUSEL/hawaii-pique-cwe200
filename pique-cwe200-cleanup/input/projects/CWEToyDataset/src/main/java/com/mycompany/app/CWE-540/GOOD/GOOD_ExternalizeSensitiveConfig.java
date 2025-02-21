import java.io.FileInputStream;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

public class GOOD_ExternalizeSensitiveConfig {

    public static void main(String[] args) {
        Properties prop = new Properties();
        Connection connection = null;
        try {
            prop.load(new FileInputStream("config/application.properties"));

            // Retrieve database configuration properties
            String databaseUrl = prop.getProperty("db.url");
            String databaseUser = prop.getProperty("db.user");
            String databasePassword = prop.getProperty("db.password");

            connection = DriverManager.getConnection(databaseUrl, databaseUser, databasePassword);
            System.out.println("Successfully connected to the database.");

            // Perform database operations...

        } catch (IOException e) {
            System.err.println("Failed to load configuration properties file.");
        } catch (SQLException e) {
            System.err.println("Database connection failed.");
        } finally {
            try {
                if (connection != null && !connection.isClosed()) {
                    connection.close();
                    System.out.println("Database connection closed.");
                }
            } catch (SQLException e) {
                System.err.println("Failed to close database connection.");
            }
        }
    }
}
