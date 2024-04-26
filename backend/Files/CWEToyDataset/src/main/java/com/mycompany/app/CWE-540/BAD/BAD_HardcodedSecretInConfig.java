import java.util.HashMap;
import java.util.Map;

public class BAD_HardcodedSecretInConfig {
    // Hardcoded sensitive information within application configuration
    private static final Map<String, String> appConfig = new HashMap<>();

    static {
        appConfig.put("apiBaseUrl", "https://api.example.com");
        appConfig.put("apiKey", "supersecretkey12345"); // Sensitive API key hardcoded
        appConfig.put("encryptionKey", "0123456789abcdef"); // Sensitive encryption key hardcoded
    }

    public static void main(String[] args) {
        // Application logic that uses appConfig for operations
        System.out.println("API Base URL: " + appConfig.get("apiBaseUrl"));
        // Using apiKey and encryptionKey in application logic would expose them to risk
    }
}
