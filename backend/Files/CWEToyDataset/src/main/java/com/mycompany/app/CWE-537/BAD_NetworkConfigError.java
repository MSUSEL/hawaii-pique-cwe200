import java.net.InetAddress;
import java.net.UnknownHostException;

public class BAD_NetworkConfigError {
    public static void main(String[] args) {
        try {
            InetAddress address = InetAddress.getByName("sensitive.internal.service");
            // Perform network operations...
        } catch (UnknownHostException e) {
            // Insecure: Detailed error about network configuration
            System.err.println("Error resolving hostname: " + e.getMessage());
        }
    }
}
