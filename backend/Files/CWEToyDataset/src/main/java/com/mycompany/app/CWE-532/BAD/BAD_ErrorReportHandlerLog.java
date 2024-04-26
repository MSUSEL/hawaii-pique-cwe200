import ch.qos.logback.classic.*;
import org.slf4j.LoggerFactory;

public class BAD_ErrorReportHandlerLog {
    private static final Logger logger = (Logger) LoggerFactory.getLogger(BAD_ErrorReportHandlerLog.class);

    public void handleError(String errorCode, String errorMessage, String debugInfo) {
        // Log detailed error information
        logger.error("Error reported: Code: {}, Message: {}, Debug Info: {}", errorCode, errorMessage, debugInfo); // Potential exposure of sensitive debug information
    }
}
