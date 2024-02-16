import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

public class BAD_ApiKeyExposureTestConfig {

    public static void main(String[] args) throws IOException {
        // Loading test configuration from a properties file that contains sensitive information
        Properties prop = new Properties();
        prop.load(new FileInputStream("test-config.properties")); // Contains API keys

        String apiKey = prop.getProperty("api.key"); // Sensitive API key exposed

        System.out.println("Using API Key: " + apiKey + " for testing");
    }
}
