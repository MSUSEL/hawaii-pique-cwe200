import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_DatabaseAccessLogger {
    private static final Logger logger = LogManager.getLogger(BAD_DatabaseAccessLogger.class);

    public static void logDatabaseAccess(String dbCommand, String connectionString) {
        try {
            Process process = Runtime.getRuntime().exec(dbCommand + " " + connectionString);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            StringBuilder errorDetails = new StringBuilder();
            String line;

            while ((line = errorReader.readLine()) != null) {
                errorDetails.append(line).append("\n");
            }

            if (process.waitFor() != 0) {
                logger.error("Database access attempt failed with connection string {}: {}", connectionString, errorDetails.toString()); // Exposing connection string and error details
            }
        } catch (IOException | InterruptedException e) {
            logger.error("Database operation failed with connection string {}: {}", connectionString, e.getMessage());
        }
    }
}
