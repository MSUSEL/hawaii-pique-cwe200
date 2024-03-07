import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class GOOD_ExternalizeSensitiveConfig {

    public static void main(String[] args) {
        Properties prop = new Properties();
        try {
            // Assuming the properties file is not included in the source code repository and is securely managed
            prop.load(new FileInputStream("config/application.properties"));

            String databaseUrl = prop.getProperty("db.url");
            String databaseUser = prop.getProperty("db.user");
            String databasePassword = prop.getProperty("db.password");

            // Use the properties to connect to the database, etc.
            System.out.println("Database URL: " + databaseUrl);
            // Further processing...
        } catch (IOException e) {
            System.err.println("An error occurred.");
        }
    }
}
