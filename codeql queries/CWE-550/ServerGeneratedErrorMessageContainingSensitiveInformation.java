import java.io.IOException;
import java.net.InetAddress;

/* Description -
Certain conditions, such as network failure, will cause a server error message to be displayed.
While error messages in and of themselves are not dangerous, per se, it is what an attacker can glean from them that might cause eventual problems.
*/
public class ServerGeneratedErrorMessageContainingSensitiveInformation {

    // Example 1 - Insecure code
    private static void insecureCheckNetworkConnection() throws IOException {
        // Simulate a network-related operation (e.g., checking connectivity to a
        // server)

        // Here we simulate a network failure, by trying to connect to a server that
        // doesn't exist.
        InetAddress serverAddress = InetAddress.getByName("CWEProject.com");
        try {
            if (!serverAddress.isReachable(5000)) {
                throw new IOException("Network connection failed. Unable to reach the server.");
            } else {
                System.out.println("Connected to sever.");
            }
        } catch (IOException e) {
            // The error message is exposed
            System.out.println("Error: " + e.getMessage() + serverAddress);
        }
    }

    // Example 2 - Secure code
    private static void secureCheckNetworkConnection() throws IOException {
        // Simulate a network-related operation (e.g., checking connectivity to a
        // server)

        // Here we simulate a network failure, by trying to connect to a server that
        // doesn't exist.
        InetAddress serverAddress = InetAddress.getByName("CWEProject.com");
        try {
            if (!serverAddress.isReachable(5000)) {
                throw new IOException("Network connection failed. Unable to reach the server.");
            } else {
                System.out.println("Connected to sever.");
            }
        } catch (IOException e) {
            // Just print a generic error message
            System.out.println("Error");
        }
    }

}
