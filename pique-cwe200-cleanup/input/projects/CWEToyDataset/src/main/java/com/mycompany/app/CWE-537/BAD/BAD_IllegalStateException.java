public class BAD_IllegalStateException {

    public void logAttempt(String username, String password, boolean success) {
        try {
            if (username == null || password == null) {
                throw new IllegalStateException();
            }
            // Continue with the login attempt logging if no exception is thrown
            System.out.println("Login attempt logged successfully for " + username);
        } catch (IllegalStateException e) {
            System.err.println("Login attempt for " + username + " with password " + password + " was " + (success ? "successful" : "unsuccessful"));
        }
    }

}
