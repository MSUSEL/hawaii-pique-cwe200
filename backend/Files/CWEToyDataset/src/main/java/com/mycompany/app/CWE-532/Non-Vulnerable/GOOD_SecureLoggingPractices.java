import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_SecureLoggingPractices {
    private static final Logger logger = LogManager.getLogger(GOOD_SecureLoggingPractices.class);

    // Method to determine if an operation is considered sensitive
    private boolean isSensitiveOperation(String operationDetails) {
        // Placeholder for logic to determine operation sensitivity
        // For example, checking if operation details contain keywords like "password", "credit card", etc.
        return operationDetails.toLowerCase().contains("password") || operationDetails.toLowerCase().contains("credit card");
    }

    public void processOperation(String operationDetails) {
        if (isSensitiveOperation(operationDetails)) {
            // For sensitive operations, log a generic message without specific details
            logger.info("Processing a sensitive operation.");
            // Additional logic to securely process the operation without exposing details in logs
        } else {
            // For non-sensitive operations, it's okay to log more specific details
            logger.info("Processing operation with details: {}", operationDetails);
            // Logic for processing non-sensitive operations
        }

        // Placeholder for operation processing logic
        // Assume this includes secure handling of any sensitive data involved
    }

    public static void main(String[] args) {
        new GOOD_SecureLoggingPractices().processOperation("Updating user password");
        new GOOD_SecureLoggingPractices().processOperation("Viewing general user profile information");
    }
}
