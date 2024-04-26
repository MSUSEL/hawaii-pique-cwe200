import java.util.*;

public class BAD_NoSuchElementException {
    public String getConfigValue(String key) {
        try {
            Properties config = loadConfig();
            return config.getProperty(key);  // Throws NoSuchElementException if the key is not found
        } catch (NoSuchElementException e) {
            System.out.println("Config key not found: " + key);
            throw new NoSuchElementException("Config key not found: " + key);
        }
    }

    private Properties loadConfig() {
        // Simulation of loading configuration properties
        Properties properties = new Properties();
        properties.put("username", "user1");
        properties.put("password", "pass123");  // Sensitive data that should not be exposed
        return properties;
    }
}
