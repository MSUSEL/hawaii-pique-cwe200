
public class BAD_IllegalStateException {

    public void logAttempt(String username, String password, boolean success) {
        try {
            if (username == null || password == null) {
                throw new IllegalStateException("Username or password cannot be null.");
            }
            System.err.println("Login attempt for " + username + " with password " + password + " was " + (success ? "successful" : "unsuccessful"));
        } catch (IllegalStateException e) {
            System.err.println("Failed to log login attempt due to missing credentials. Error: " + e.getMessage());
            throw new IllegalStateException("Logging error: " + e.getMessage());
        }
    }
}
