import java.util.logging.*;

public class BAD_IllegalStateException {
    private static final Logger logger = Logger.getLogger("SecurityLogger");

    public void logAttempt(String username, String password, boolean success) {
        try {
            if (username == null || password == null) {
                throw new IllegalStateException("Username or password cannot be null.");
            }
            logger.info("Login attempt for " + username + " with password " + password + " was " + (success ? "successful" : "unsuccessful"));
        } catch (IllegalStateException e) {
            System.err.println("Failed to log login attempt due to missing credentials. Error: " + e.getMessage());
            throw new IllegalStateException("Logging error: " + e.getMessage());
        }
    }
}
