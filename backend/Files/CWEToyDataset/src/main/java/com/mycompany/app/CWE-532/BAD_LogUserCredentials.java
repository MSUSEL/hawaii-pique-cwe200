import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class BAD_LogUserCredentials {
    private static final Logger logger = LogManager.getLogger(BAD_LogUserCredentials.class);

    public void loginUser(String username, String password) {
        // Insecure logging of sensitive information
        logger.info("Attempting login for username: " + username + " with password: " + password);

        // Authentication logic...
    }

    public static void main(String[] args) {
        new BAD_LogUserCredentials().loginUser("user1", "pass1234");
    }
}
