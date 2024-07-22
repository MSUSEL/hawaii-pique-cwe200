import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BAD_ConfigLoaderTest {
    @Test
    public void testLoadConfig() {
        ConfigLoader loader = new ConfigLoader();
        String apiKey = "ABCD-1234-EFGH-5678"; 
        assertEquals(loader.loadAPIKey(), apiKey, "API keys should match.");
    }

    class ConfigLoader {
        public String loadAPIKey() {
            return "ABCD-1234-EFGH-5678"; 
        }
    }
}
