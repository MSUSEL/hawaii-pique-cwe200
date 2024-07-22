import java.util.HashMap;
import java.util.Map;

public class BAD_HardcodedSecretInConfig {
    private static final Map<String, String> appConfig = new HashMap<>();

    static {
        appConfig.put("apiBaseUrl", "https://api.example.com");
        appConfig.put("apiKey", "supersecretkey12345");
        appConfig.put("encryptionKey", "0123456789abcdef"); 
    }

    public static void main(String[] args) {
        // Application logic that uses appConfig for operations
        System.out.println("API Base URL: " + appConfig.get("apiBaseUrl"));
    }
}
