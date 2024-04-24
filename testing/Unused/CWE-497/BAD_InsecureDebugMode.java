import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BAD_InsecureDebugMode {
    private static final Logger logger = LoggerFactory.getLogger(BAD_InsecureDebugMode.class);
    private static final boolean DEBUG_MODE = true; // Flag to control debug mode

    public static void main(String[] args) {
        try {
            // Simulated sensitive operation
            performSensitiveOperation();
        } catch (Exception e) {
            if (DEBUG_MODE) {
                // Insecure: Logging exception details including stack trace when in debug mode
                logger.error("Exception occurred during sensitive operation: " + e.getMessage(), e);
                e.printStackTrace(); // Even more dangerous if this is a web application
            }
        }
    }

    private static void performSensitiveOperation() throws Exception {
        // Placeholder for an operation that might fail
        throw new Exception("Sensitive operation failed due to configuration error: /etc/app/config.secret");
    }
}
