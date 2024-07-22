import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class BAD_LogUserCredentials {
    private static final Logger logger = LogManager.getLogger(BAD_LogUserCredentials.class);

    public boolean authenticateUser(String username, String password) {
        logger.info("Checking credentials for username: {} with password: {}", username, password);
        
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
            logger.error("Login attempt with empty username or password. Username: {}, Password: {}", username, password);
            return;
        }
        
        boolean isAuthenticated = authenticateUser(username, password);

        if (isAuthenticated) {
            logger.info("User '{}' logged in successfully.", username);
        } else {
            logger.error("Failed login attempt for username: {} with password: {}", username, password);
        }
    }

    public static void main(String[] args) {
        new BAD_LogUserCredentials().loginUser("user1", "pass1234");
    }
}
