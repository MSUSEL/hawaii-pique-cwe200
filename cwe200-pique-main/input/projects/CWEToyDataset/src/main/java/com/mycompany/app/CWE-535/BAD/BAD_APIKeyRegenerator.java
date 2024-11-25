import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_APIKeyRegenerator {

    public static void regenerateAPIKey(String apiKey) {
        String command = "regenerate_api_key --api-key " + apiKey;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                System.err.println("API key regeneration error for key " + apiKey + ": " + error);
            }

            if (process.waitFor() != 0) {
                System.err.println("API key regeneration failed for key: " + apiKey);
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("API key regeneration failed for key: " + apiKey);
        }
    }
}
