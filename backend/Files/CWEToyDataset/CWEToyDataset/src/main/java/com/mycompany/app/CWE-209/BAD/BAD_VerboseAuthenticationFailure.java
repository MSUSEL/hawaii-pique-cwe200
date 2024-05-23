public class BAD_VerboseAuthenticationFailure {
    // This class shows an insecure approach to handling authentication failures by providing detailed error messages.

    public static void main(String[] args) {
        String username = "user";
        String password = "guessme";

        if (!authenticate(username, password)) {
            // Insecure: Revealing the reason for authentication failure might expose user existence or password correctness.
            System.out.println("Authentication failed: " + username + " does not exist or password is incorrect.");
        }
    }

    private static boolean authenticate(String username, String password) {
        // Placeholder for authentication logic
        // Assume authentication fails for demonstration purposes
        return false;
    }
}
