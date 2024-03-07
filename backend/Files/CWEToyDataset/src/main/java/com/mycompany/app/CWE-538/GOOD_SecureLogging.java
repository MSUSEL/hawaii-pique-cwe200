import java.io.FileWriter;
import java.io.IOException;

public class GOOD_SecureLogging {
    public static void main(String[] args) {
        String username = "admin";

        try (FileWriter logWriter = new FileWriter("/var/log/myapp/access.log", true)) {
            // Secure: Logging without including sensitive information
            logWriter.write("User login attempt: " + username + "\n");
        } catch (IOException e) {
            System.err.println("An error occurred.");
        }
    }
}
