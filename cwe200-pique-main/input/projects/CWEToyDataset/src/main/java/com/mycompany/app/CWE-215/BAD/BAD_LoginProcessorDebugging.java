public class BAD_LoginProcessorDebugging {
    private static final boolean DEBUG_ON_ERROR = Boolean.parseBoolean(System.getenv("DEBUG_ON_ERROR"));

    public static void main(String[] args) {
        try {
            authenticateUser("username", "password");
        } catch (AuthenticationException e) {
            if (DEBUG_ON_ERROR) {
                System.err.println("DEBUG: Authentication failed for user: " + e.getUsername() + " with error: " + e.getMessage());
            }
        }
    }

    private static void authenticateUser(String username, String password) throws AuthenticationException {
        // Simulate an authentication check
        throw new AuthenticationException("Invalid credentials", username);
    }

    private static class AuthenticationException extends Exception {
        private String username;

        public AuthenticationException(String message, String username) {
            super(message);
            this.username = username;
        }

        public String getUsername() {
            return username;
        }
    }
}
