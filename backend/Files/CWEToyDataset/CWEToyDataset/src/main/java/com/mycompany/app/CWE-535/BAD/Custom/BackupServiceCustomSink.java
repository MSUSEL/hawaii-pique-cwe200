import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class BackupServiceCustomSink {

    public static void performBackup(String backupLocation) {
        String command = "perform_backup --location " + backupLocation;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                OutputToDisplay.output("Backup error: " + error);
            }

            if (process.waitFor() != 0) {
                OutputToDisplay.output("Backup failed, see logs for details.");
            }
        } catch (IOException | InterruptedException e) {
            OutputToDisplay.output("Backup operation failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        performBackup("/secure/backup/location");
    }
}
