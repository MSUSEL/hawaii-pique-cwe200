import java.io.*;
import org.apache.logging.log4j.*;

public class GOOD_DatabaseAccessLogger {
    private static final Logger logger = LogManager.getLogger(GOOD_DatabaseAccessLogger.class);

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
                logger.error("Database access attempt failed with connection string.");
            }
        } catch (IOException | InterruptedException e) {
            logger.error("Database operation failed.");
        }
    }
}
