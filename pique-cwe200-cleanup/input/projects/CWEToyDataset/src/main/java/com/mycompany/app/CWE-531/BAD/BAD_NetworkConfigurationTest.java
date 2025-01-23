import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BAD_NetworkConfigurationTest {
    @Test
    public void testConfigureNetwork() {
        String deviceSerial = "SN12345678";
        String configCommands = "set admin-password = 'newPassword123!';";
        NetworkConfiguration config = new NetworkConfiguration();
        assertTrue(config.configureNetwork(deviceSerial, configCommands));
    }

    class NetworkConfiguration {
        public boolean configureNetwork(String deviceSerial, String configCommands) {
            // Pretend to configure network devices with sensitive commands
            return !configCommands.isEmpty();
        }
    }
}
