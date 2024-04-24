import java.util.logging.*;

public class GOOD_IllegalStateException {
    private static final Logger logger = Logger.getLogger("SecurityLogger");

    public void logAttempt(String username, String password, boolean success) {
        try {
            if (username == null || password == null) {
                throw new IllegalStateException("Username or password cannot be null.");
            }
            logger.info("Login attempt.");
        } catch (IllegalStateException e) {
            System.err.println("Failed to login attempt due to missing credentials.");
            throw new IllegalStateException("Logging error.");
        }
    }
}
