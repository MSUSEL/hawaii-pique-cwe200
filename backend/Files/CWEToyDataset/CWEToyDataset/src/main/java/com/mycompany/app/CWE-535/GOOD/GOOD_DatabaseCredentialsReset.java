import java.io.*;

public class GOOD_DatabaseCredentialsReset {
    public static void resetDBPassword(String dbCommand, String username, String newPassword) {
        try {
            String command = dbCommand + " --user " + username + " --password " + newPassword;
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorOutput = errorReader.readLine();

            if (process.waitFor() != 0) {
                System.err.println("Failed to reset database password.");
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error resetting database credentials.");
        }
    }
}
