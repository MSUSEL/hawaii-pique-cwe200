import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class BAD_NetworkConfiguratorCustomSink {

    public static void configureNetwork(String networkConfig) {
        String command = "configure_network --config " + networkConfig;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                DisplayOnScreen.display("Network configuration error: " + error);
            }

            if (process.waitFor() != 0) {
                DisplayOnScreen.display("Network configuration failed, see logs for details.");
            }
        } catch (IOException | InterruptedException e) {
            DisplayOnScreen.display("Configuration operation failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        configureNetwork("{ \"ip\": \"192.168.1.1\", \"gateway\": \"192.168.1.254\" }");
    }
}
