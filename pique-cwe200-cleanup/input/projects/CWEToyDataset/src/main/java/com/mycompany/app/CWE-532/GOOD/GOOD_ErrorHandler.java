import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class GOOD_ErrorHandler {
    private static final Logger logger = LogManager.getLogger(GOOD_ErrorHandler.class);

    public void logError(String errorMessage, String userSession) {
        logger.error("An error occurred.");
    }
}
