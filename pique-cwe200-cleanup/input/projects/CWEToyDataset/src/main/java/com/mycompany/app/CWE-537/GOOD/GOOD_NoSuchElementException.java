import java.util.*;

public class GOOD_NoSuchElementException {
    public String getConfigValue(String key) {
        try {
            Properties config = loadConfig();
            return config.getProperty(key);  // Throws NoSuchElementException if the key is not found
        } catch (NoSuchElementException e) {
            System.out.println("Config key not found.");
            throw new NoSuchElementException("Config key not found");
        }
    }

    private Properties loadConfig() {
        // Simulation of loading configuration properties
        Properties properties = new Properties();
        properties.put("username", "user1");
        properties.put("password", "pass123"); 
        return properties;
    }
}
