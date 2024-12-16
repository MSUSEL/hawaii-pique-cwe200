import java.io.BufferedReader;
import java.io.InputStreamReader;

public class GOOD_SecureShellCommandHandling {

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "cp /path/to/sensitive/file /backup/location"};
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                logErrorSecurely("Backup operation failed. Please check system logs.");
            }
        } catch (Exception e) {
            logErrorSecurely("Exception in executing backup command.");
        }
    }

    private static void logErrorSecurely(String message) {
        System.out.println(message);
    }
}
