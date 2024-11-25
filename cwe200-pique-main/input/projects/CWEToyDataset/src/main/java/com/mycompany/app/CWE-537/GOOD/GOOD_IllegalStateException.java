public class GOOD_IllegalStateException {

    public void logAttempt(String username, String password, boolean success) {
        try {
            if (username == null || password == null) {
                throw new IllegalStateException("Username or password cannot be null.");
            }
        } catch (IllegalStateException e) {
            System.err.println("Failed to login attempt due to missing credentials.");
            throw new IllegalStateException("Logging error.");
        }
    }
}
