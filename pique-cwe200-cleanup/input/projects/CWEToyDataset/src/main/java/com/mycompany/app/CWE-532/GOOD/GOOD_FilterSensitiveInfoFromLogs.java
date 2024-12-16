import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_FilterSensitiveInfoFromLogs {
    private static final Logger logger = LogManager.getLogger(GOOD_FilterSensitiveInfoFromLogs.class);

    private boolean isUsernameValid(String username) {
        // Placeholder for actual validation logic
        return username != null && !username.trim().isEmpty();
    }

    private boolean doesUserExist(String username) {
        // Placeholder for user existence check. Always returns true for demonstration.
        return true; // Assume user exists for this example
    }

    public void loginUser(String username, String password) {
        if (!isUsernameValid(username)) {
            logger.warn("Login attempt with invalid username format.");
            return;
        }

        if (!doesUserExist(username)) {
            logger.info("Login attempt for non-existing user.");
            return;
        }

        boolean isAuthenticated = false; // Placeholder for the authentication result


        if (isAuthenticated) {
            logger.info("User logged in successfully.");
        } else {
            logger.warn("Failed login attempt due to incorrect credentials.");
        }
    }

    public static void main(String[] args) {
        new GOOD_FilterSensitiveInfoFromLogs().loginUser("user1", "securePassword");
    }
}
