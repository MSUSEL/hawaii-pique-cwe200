import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class BAD_StoreDbConnectionInfo {
    public static void main(String[] args) {
        Properties prop = new Properties();
        prop.setProperty("dbUrl", "jdbc:mysql://localhost:3306/sensitiveDB");
        prop.setProperty("dbUser", "sensitiveUser");
        prop.setProperty("dbPassword", "sensitivePassword"); // Sensitive Information

        try (FileOutputStream output = new FileOutputStream("/tmp/dbConfig.properties")) {
            // Insecure: Storing sensitive DB connection details in a world-readable file
            prop.store(output, "Database connection details");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
