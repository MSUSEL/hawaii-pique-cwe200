import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

class NetworkConfiguration {
    public boolean configureNetwork(String deviceSerial, String configCommands) {
        // Pretend to configure network devices with sensitive commands
        return !configCommands.isEmpty();
    }
}

public class GOOD_NetworkConfigurationTest {
    private Properties testProps;

    @BeforeEach
    public void loadTestProperties() {
        testProps = new Properties();
        try (FileInputStream in = new FileInputStream("test_network_config.properties")) {
            testProps.load(in);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load network configuration properties", e);
        }
    }

    @Test
    public void testConfigureNetwork() {
        String deviceSerial = testProps.getProperty("deviceSerial");
        String configCommands = testProps.getProperty("configCommands");

        NetworkConfiguration config = new NetworkConfiguration();
        assertTrue(config.configureNetwork(deviceSerial, configCommands), "Network should be configured successfully.");
    }
}
