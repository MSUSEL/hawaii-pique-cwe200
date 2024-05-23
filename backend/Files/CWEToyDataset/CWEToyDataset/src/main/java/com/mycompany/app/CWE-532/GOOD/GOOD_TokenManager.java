import org.slf4j.*;

public class GOOD_TokenManager {
    private static final Logger logger = LoggerFactory.getLogger(GOOD_TokenManager.class);

    public void refreshToken(String userId, String oldToken) {
        // Simulate token refresh process
        String newToken = "newToken12345"; // Simulated new token
        logger.debug("Refreshing token."); // Insecure logging of tokens
    }
}
