public class GOOD_SecureDebuggingPractices {
    private static final boolean DEBUG_ENABLED = false; // Debugging flag, should be false in production

    public static void main(String[] args) {
        String username = "user";
        String password = "password123"; // Sensitive information

        if (DEBUG_ENABLED) {
            // Debugging code is gated by a flag, ensuring it's disabled in production environments
            System.out.println("DEBUG: Attempting login with username: " + username);
            // Note: Sensitive details like passwords should never be logged, even in debugging mode
        }
        
        performLogin(username, password);
    }

    private static void performLogin(String username, String password) {
        // Placeholder for actual login logic
    }
}
