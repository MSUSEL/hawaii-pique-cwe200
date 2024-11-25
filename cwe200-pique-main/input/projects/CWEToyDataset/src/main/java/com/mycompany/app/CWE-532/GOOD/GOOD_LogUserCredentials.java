import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class GOOD_LogUserCredentials {
    private static final Logger logger = LogManager.getLogger(GOOD_LogUserCredentials.class);

    public boolean authenticateUser(String username, String password) {
        logger.info("Checking credentials.");
        
        // Placeholder for database lookup that might involve insecure practices
        // This part is purely illustrative and skips actual database operations
        if ("admin".equals(username) && "adminPass".equals(password)) {
            return true; // Simulated successful authentication
        } else if ("user1".equals(username) && "pass1234".equals(password)) {
            return true; // Another simulated successful authentication
        }
        return false; // Default to authentication failure
    }

    public void loginUser(String username, String password) {
        if (username == null || password == null || username.isEmpty() || password.isEmpty()) {
            logger.error("Login attempt with empty username or password.");
            return;
        }
        
        boolean isAuthenticated = authenticateUser(username, password);

        if (isAuthenticated) {
            logger.info("User logged in successfully.");
        } else {
            logger.error("Failed login attempt.");
        }
    }

    public static void main(String[] args) {
        new GOOD_LogUserCredentials().loginUser("user1", "pass1234");
    }
}
