import ch.qos.logback.classic.*;

public class BAD_ErrorReportHandler {
    private static final Logger logger = (Logger) LoggerFactory.getLogger(BAD_ErrorReportHandler.class);

    public void handleError(String errorCode, String errorMessage, String debugInfo) {
        // Log detailed error information
        logger.error("Error reported: Code: {}, Message: {}, Debug Info: {}", errorCode, errorMessage, debugInfo); // Potential exposure of sensitive debug information
    }
}
