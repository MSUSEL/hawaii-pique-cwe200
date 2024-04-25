import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NetworkConfiguration {
    public boolean configureNetwork(String deviceSerial, String configCommands) {
        // Pretend to configure network devices with sensitive commands
        return !configCommands.isEmpty();
    }
}

public class BAD_NetworkConfiguration {
    @Test
    public void testConfigureNetwork() {
        String deviceSerial = "SN12345678";
        String configCommands = "set admin-password = 'newPassword123!';"; // Sensitive command in test
        NetworkConfiguration config = new NetworkConfiguration();
        assertTrue(config.configureNetwork(deviceSerial, configCommands));
    }
}
