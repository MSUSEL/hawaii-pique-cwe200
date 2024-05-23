import java.io.FileWriter;
import java.io.IOException;

public class BAD_LoggingCredentials {
    public static void main(String[] args) {
        String username = "admin";
        String password = "password123"; // Sensitive Information

        try (FileWriter logWriter = new FileWriter("/var/www/html/logs/access.log", true)) {
            // Insecure: Writing sensitive information to an externally accessible log file
            logWriter.write("User login attempt: " + username + " with password " + password + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
