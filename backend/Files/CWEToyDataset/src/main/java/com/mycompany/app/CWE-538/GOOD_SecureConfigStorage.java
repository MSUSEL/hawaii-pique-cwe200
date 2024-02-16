import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class GOOD_SecureConfigStorage {
    public static void main(String[] args) {
        Properties prop = new Properties();
        prop.setProperty("dbUrl", "jdbc:mysql://localhost:3306/secureDB");

        try (FileOutputStream output = new FileOutputStream("/etc/myapp/dbConfig.secure")) {
            // Secure: Storing configuration details without sensitive information in a secure location
            prop.store(output, "Secure Database connection details");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
