import java.util.logging.Logger;

public class BAD_LoggingSensitiveInformation {
    private static final Logger LOGGER = Logger.getLogger(BAD_LoggingSensitiveInformation.class.getName());
    // Demonstrates an insecure practice of logging sensitive information.

    public static void main(String[] args) {
        String username = System.getenv("username");
        String password = System.getenv("password");

        // Insecure: Logging sensitive information that could be exposed in log files
        LOGGER.info("Attempting login with username: " + username + " and password: " + password);
    }
}
