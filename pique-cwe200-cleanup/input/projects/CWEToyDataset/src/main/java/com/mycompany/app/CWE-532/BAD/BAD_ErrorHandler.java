import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class BAD_ErrorHandler {
    private static final Logger logger = LogManager.getLogger(BAD_ErrorHandler.class);

    public void logError(String errorMessage, String userSession) {
        logger.error("An error occurred: " + errorMessage + ". Session: " + userSession);
    }
}
