import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_SecureLoggingPractices {
    private static final Logger logger = LogManager.getLogger(GOOD_SecureLoggingPractices.class);

    public void processSensitiveOperation(String operationDetails) {
        // Only log non-sensitive operation details or use placeholders for sensitive data
        logger.info("Processing sensitive operation with details: {}", operationDetails.replaceAll("[\\s\\S]", "*"));

        // Sensitive operation logic...
    }

    public static void main(String[] args) {
        new GOOD_SecureLoggingPractices().processSensitiveOperation("Sensitive operation details here");
    }
}
