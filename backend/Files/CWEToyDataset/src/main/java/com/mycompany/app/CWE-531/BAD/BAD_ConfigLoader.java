import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;


class ConfigLoader {
    public String loadAPIKey() {
        return "ABCD-1234-EFGH-5678"; // Return a mocked API key
    }
}

public class BAD_ConfigLoader {
    @Test
    public void testLoadConfig() {
        ConfigLoader loader = new ConfigLoader();
        String apiKey = "ABCD-1234-EFGH-5678"; // Exposed API key
        assertEquals(loader.loadAPIKey(), apiKey, "API keys should match.");
    }
}
