import java.util.logging.Logger;

public class GOOD_MinimalDataLogging{
    private static final Logger LOGGER = Logger.getLogger(GOOD_MinimalDataLogging.class.getName());
    // Demonstrates secure logging practices that avoid logging sensitive information.

    public static void main(String[] args) {
        String username = "admin";
        // Secure: Avoid logging sensitive information. Log actions without specific details.

        LOGGER.info("Attempting login for username: " + username);
        // Note: The password or other sensitive information is not logged
    }
}
