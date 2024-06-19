import java.util.*;

public class BAD_NoSuchElementException {
    public String getConfigValue(String key) {
        Properties config = loadConfig();

        if (config.getProperty(key) == null) {
            System.out.println("Config key not found: " + key);
            throw new NoSuchElementException("Config key not found: " + key);
        }
        else{
            return config.getProperty(key);
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
