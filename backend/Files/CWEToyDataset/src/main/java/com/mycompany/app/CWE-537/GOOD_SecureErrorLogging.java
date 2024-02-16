import java.util.logging.Level;
import java.util.logging.Logger;

public class GOOD_SecureErrorLogging {
    private static final Logger LOGGER = Logger.getLogger(GOOD_SecureErrorLogging.class.getName());

    public static void main(String[] args) {
        try {
            // Sensitive operation that might fail
            performSensitiveOperation();
        } catch (Exception e) {
            // Secure: Log the detailed error for internal diagnosis and provide a generic error message externally
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            System.err.println("An error occurred. Please contact technical support.");
        }
    }

    private static void performSensitiveOperation() throws Exception {
        // Simulate an operation that can fail
        throw new Exception("Sensitive operation failed due to misconfiguration.");
    }
}
