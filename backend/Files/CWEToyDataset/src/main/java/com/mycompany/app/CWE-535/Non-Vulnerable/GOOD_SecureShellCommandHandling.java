import java.io.BufferedReader;
import java.io.InputStreamReader;

public class GOOD_SecureShellCommandHandling {
    // Illustrates secure handling of shell command execution to prevent exposure of sensitive information.

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "cp /path/to/sensitive/file /backup/location"};
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                // Secure: Logging error to a secure, controlled location instead of exposing it directly.
                logErrorSecurely("Backup operation failed. Please check system logs.");
            }
        } catch (Exception e) {
            logErrorSecurely("Exception in executing backup command.");
        }
    }

    private static void logErrorSecurely(String message) {
        // Placeholder for secure logging mechanism
        // In practice, this should log to a file or system that is securely accessible only to authorized personnel.
        System.out.println(message);
    }
}
