import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ConfigLoader {
    public String loadAPIKey() {
        // Ideally, this method should fetch the API key from a secure source or environment variable
        return System.getenv("API_KEY"); // Fetch API key from environment variables
    }
}

public class GOOD_ConfigLoaderTest {
    @Test
    public void testLoadConfig() {
        ConfigLoader loader = new ConfigLoader();
        // The API key is no longer hardcoded here, it's fetched from the environment during runtime
        String expectedApiKey = System.getenv("API_KEY"); // Fetch expected API key from environment variables
        String actualApiKey = loader.loadAPIKey();
        
        assertEquals(expectedApiKey, actualApiKey, "API keys should match.");
    }
}
