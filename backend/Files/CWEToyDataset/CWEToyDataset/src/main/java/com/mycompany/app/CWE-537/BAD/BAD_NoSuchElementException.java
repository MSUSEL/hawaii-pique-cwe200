import java.util.*;

public class BAD_NoSuchElementException {
    public String getConfigValue(String key) {
        Properties config = loadConfig();

        try {
            if (config.getProperty(key) == null) {
                throw new NoSuchElementException();
            } else {
                return config.getProperty(key);
            }
        } catch (NoSuchElementException e) {
            System.err.println("Config key not found: " + key);
            return null;
        }
    }

    private Properties loadConfig() {
        // Simulation of loading configuration properties
        Properties properties = new Properties();
        properties.put("username", "user1");
        properties.put("password", "pass123"); 
        return properties;
    }

    public static void main(String[] args) {
        BAD_NoSuchElementException example = new BAD_NoSuchElementException();

        System.out.println("Username: " + example.getConfigValue("username"));
        System.out.println("Password: " + example.getConfigValue("password"));
        System.out.println("Non-existent key: " + example.getConfigValue("nonExistentKey"));
    }
}
