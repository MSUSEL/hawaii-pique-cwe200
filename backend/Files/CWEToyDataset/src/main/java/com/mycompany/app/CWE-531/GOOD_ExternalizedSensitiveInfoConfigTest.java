import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

public class GOOD_ExternalizedSensitiveInfoConfigTest {

    public static void main(String[] args) throws IOException {
        // Loading non-sensitive test configuration from a properties file
        Properties prop = new Properties();
        prop.load(new FileInputStream("test-config.properties"));

        // Sensitive data is fetched from environment variables, not stored in the file
        String apiKey = System.getenv("API_KEY");

        System.out.println("Using securely fetched API Key for testing");
    }
}
