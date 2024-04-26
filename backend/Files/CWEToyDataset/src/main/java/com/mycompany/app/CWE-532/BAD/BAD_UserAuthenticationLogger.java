import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BAD_UserAuthenticationLogger {
    private static final Logger logger = LoggerFactory.getLogger(BAD_UserAuthenticationLogger.class);

    public void authenticateUser(String username, String password) {
        // Insecurely logging username and password
        logger.debug("Attempting to authenticate user: " + username + " with password: " + password);
    }
}
