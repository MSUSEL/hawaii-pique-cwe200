// This example introduces a consistent timing delay for authentication as to not exposing if a user exists or not.
public class GOOD_ConsistentAuthenticationTiming {
    
    // Fetching the valid username and password from environment variables for demonstration purposes
    private static final String VALID_USERNAME = System.getenv("VALID_USERNAME");
    private static final String VALID_PASSWORD = System.getenv("VALID_PASSWORD");

    public static boolean consistentAuthenticate(String attemptUsername, String attemptPassword) {
        boolean validUser = attemptUsername.equals(VALID_USERNAME);
        boolean correctPassword = attemptPassword.equals(VALID_PASSWORD);

        // Introducing a consistent delay for all authentication attempts
        try {
            Thread.sleep(1000); // Fixed delay to mask timing discrepancies
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return validUser && correctPassword;
    }

    public static void main(String[] args) {
        // Example usage; in a real scenario, user input would be handled securely and not hardcoded
        consistentAuthenticate(System.getenv("ATTEMPT_USERNAME"), System.getenv("ATTEMPT_PASSWORD"));
    }
}
