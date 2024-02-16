public class BAD_DebugOutputWithCredentials {
    // Exposes the risky practice of including user credentials directly in debug output.

    public static void authenticateUser(String username, String password) {
        // Placeholder authentication logic
        boolean isAuthenticated = false;

        if (!isAuthenticated) {
            // Insecure: User credentials are included in the debug output
            System.out.println("DEBUG: Authentication failed for user: " + username + " with password: " + password);
        }
    }

    public static void main(String[] args) {
        authenticateUser("admin", "adminSecret");
    }
}
