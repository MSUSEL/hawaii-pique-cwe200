import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GOOD_SecureErrorLogging2 {
    private static final Logger logger = LoggerFactory.getLogger(GOOD_SecureErrorLogging2.class);

    public static void main(String[] args) {
        try {
            // Some sensitive operation
            throw new Exception("Sensitive database connection failure");
        } catch (Exception e) {
            // Secure: Logging the detailed error internally while showing a generic message to the user
            logger.error("An internal error occurred", e);
            System.out.println("An error occurred. Please contact support if the problem persists.");
        }
    }
}
