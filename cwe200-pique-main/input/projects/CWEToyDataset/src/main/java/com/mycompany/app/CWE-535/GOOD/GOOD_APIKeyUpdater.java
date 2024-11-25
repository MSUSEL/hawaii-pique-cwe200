import java.io.*;

public class GOOD_APIKeyUpdater {
    public static void updateAPIKey(String updateCommand, String apiKey) {
        try {
            Process process = Runtime.getRuntime().exec(updateCommand + " " + apiKey);
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                System.err.println("API Key update failed.");
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error updating API key.");
        }
    }
}
