import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

public class GOOD_GenericErrorMessages {
    private static final Logger LOGGER = Logger.getLogger(GOOD_GenericErrorMessages.class.getName());

    public static void main(String[] args) {
        try {
            // Load sensitive details securely from an external source, such as environment variables or a properties file
            Properties prop = loadProperties("config.properties");
            String dbUrl = prop.getProperty("databaseUrl");
            String dbUser = prop.getProperty("databaseUser");
            String dbPassword = prop.getProperty("databasePassword");
            
            connectToDatabase(dbUrl, dbUser, dbPassword);
            System.out.println("Operation was successful.");
        } catch (Exception e) {
            // Log the error for internal use with specific details
            LOGGER.log(Level.SEVERE, "An error occurred during the operation.");
            // Provide a generic error message to the user
            System.out.println("An error occurred. Please contact support.");
        }
    }

    private static Properties loadProperties(String fileName) throws IOException {
        FileInputStream fis = null;
        Properties prop = null;
        
        try {
            fis = new FileInputStream(fileName);
            prop = new Properties();
            prop.load(fis);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Unable to load configuration file.");
            throw e;
        } finally {
            if (fis != null) {
                fis.close();
            }
        }
        
        return prop;
    }

    private static void connectToDatabase(String url, String user, String password) {
        // Placeholder for database connection logic that might throw exceptions
        // For the sake of example, let's simulate a connection issue
        throw new RuntimeException("Failed to connect to the database.");
    }
}
