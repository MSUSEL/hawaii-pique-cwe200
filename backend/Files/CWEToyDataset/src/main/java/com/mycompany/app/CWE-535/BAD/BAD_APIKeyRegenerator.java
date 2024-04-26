import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_APIKeyRegenerator {
    private static final Logger logger = LogManager.getLogger(BAD_APIKeyRegenerator.class);

    public static void regenerateAPIKey(String apiKey) {
        String command = "regenerate_api_key --api-key " + apiKey;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                logger.warn("API key regeneration error for key {}: {}", apiKey, error); // Exposing API key and error details
            }

            if (process.waitFor() != 0) {
                logger.error("API key regeneration failed, see logs for key: {}", apiKey);
            }
        } catch (IOException | InterruptedException e) {
            logger.error("Regeneration operation failed for API key {}: {}", apiKey, e.getMessage());
        }
    }
}
