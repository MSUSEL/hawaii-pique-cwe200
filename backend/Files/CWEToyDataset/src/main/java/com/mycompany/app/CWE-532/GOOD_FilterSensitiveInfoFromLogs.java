import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_FilterSensitiveInfoFromLogs {
    private static final Logger logger = LogManager.getLogger(GOOD_FilterSensitiveInfoFromLogs.class);

    public void loginUser(String username) {
        // Secure logging without sensitive information
        logger.info("Attempting login for username: " + username);

        // Authentication logic...
    }

    public static void main(String[] args) {
        new GOOD_FilterSensitiveInfoFromLogs().loginUser("user1");
    }
}
