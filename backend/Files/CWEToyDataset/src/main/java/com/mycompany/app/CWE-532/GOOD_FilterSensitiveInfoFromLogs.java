import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_FilterSensitiveInfoFromLogs {
    private static final Logger logger = LogManager.getLogger(GOOD_FilterSensitiveInfoFromLogs.class);

    // Simulated method to validate username format without revealing details
    private boolean isUsernameValid(String username) {
        // Placeholder for actual validation logic
        return username != null && !username.trim().isEmpty();
    }

    // Simulated method to check if the username exists in the system without revealing details
    private boolean doesUserExist(String username) {
        // Placeholder for user existence check. Always returns true for demonstration.
        return true; // Assume user exists for this example
    }

    public void loginUser(String username, String password) {
        if (!isUsernameValid(username)) {
            logger.warn("Login attempt with invalid username format.");
            // Return or handle invalid username format without exposing the username
            return;
        }

        if (!doesUserExist(username)) {
            logger.info("Login attempt for non-existing user.");
            // Return or handle non-existing user without exposing the username
            return;
        }

        // Proceed with authentication, assuming password check and other security measures are in place
        boolean isAuthenticated = false; // Placeholder for the authentication result

        // Note: The actual authentication process, including password verification, is not detailed here.
        //       Assume it's securely handled without logging sensitive details.

        if (isAuthenticated) {
            logger.info("User logged in successfully.");
        } else {
            // Log failed login without exposing user details
            logger.warn("Failed login attempt due to incorrect credentials.");
        }
    }

    public static void main(String[] args) {
        new GOOD_FilterSensitiveInfoFromLogs().loginUser("user1", "securePassword");
    }
}
