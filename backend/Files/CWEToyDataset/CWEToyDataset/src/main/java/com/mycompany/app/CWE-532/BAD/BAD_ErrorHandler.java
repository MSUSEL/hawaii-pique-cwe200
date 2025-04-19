import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class BAD_ErrorHandler {
    private static final Logger logger = LogManager.getLogger(BAD_ErrorHandler.class);

    /**
     * This method logs not only the error message and session ID,
     * but also the user’s password, authentication token, and
     * even includes the full exception stack trace — all of which
     * are highly sensitive.
     */
    public void logError(
            String errorMessage,
            String userSession,
            String userPassword,
            String authToken,
            String creditCardNumber,
            Exception exception
    ) {
        // Build a single log line that leaks everything
        StringBuilder sb = new StringBuilder();
        sb.append("An error occurred: ").append(errorMessage)
          .append(" | Session ID: ").append(userSession)
          .append(" | Password: ").append(userPassword)
          .append(" | AuthToken: ").append(authToken)
          .append(" | CC#: ").append(creditCardNumber);

        // Logging the combined sensitive data *and* the full stack trace
        logger.error(sb.toString(), exception);
    }
}
