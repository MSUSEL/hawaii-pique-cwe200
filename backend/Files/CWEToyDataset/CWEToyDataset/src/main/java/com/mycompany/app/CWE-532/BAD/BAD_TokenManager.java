import org.slf4j.*;

public class BAD_TokenManager {
    private static final Logger logger = LoggerFactory.getLogger(BAD_TokenManager.class);

    public void refreshToken(String userId, String oldToken) {
        // Simulate token refresh process
        String newToken = "newToken12345"; // Simulated new token
        logger.debug("Refreshing token for user: {}, old token: {}, new token: {}", userId, oldToken, newToken);
    }
}
