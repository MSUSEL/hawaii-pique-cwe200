import ch.qos.logback.classic.*;
import org.slf4j.LoggerFactory;


public class GOOD_ErrorReportHandler {
    private static final Logger logger = (Logger) LoggerFactory.getLogger(GOOD_ErrorReportHandler.class);

    public void handleError(String errorCode, String errorMessage, String debugInfo) {
        logger.error("Error occured"); 
    }
}
