// Define a custom checked exception
class AuthenticationFailureException extends Exception {
    public AuthenticationFailureException(String message) {
        super(message);
    }
}

public class BAD_VerboseAuthenticationFailure {

    public static void main(String[] args) {
        String username = "user";
        String password = "guessme";

        try {
            authenticate(username, password);
            System.out.println("Authentication successful.");
        } catch (AuthenticationFailureException e) {
            logError(e.getMessage());
        }
    }

    private static void authenticate(String username, String password) throws AuthenticationFailureException {
        // Simulated authentication logic
        if (!isValid(username, password)) {
            throw new AuthenticationFailureException("Authentication failed for user: " + username + "with password " + password + ". Invalid credentials.");
        }
    }

    private static boolean isValid(String username, String password) {
        // Placeholder for actual authentication logic
        return false;
    }

    private static void logError(String errorMessage) {
        // Securely log the error
        System.err.println("Authentication Error: " + errorMessage); // Replace with a secure logging framework if needed
    }
}
