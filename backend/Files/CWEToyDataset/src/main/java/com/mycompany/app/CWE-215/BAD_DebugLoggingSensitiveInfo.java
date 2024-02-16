public class BAD_DebugLoggingSensitiveInfo {
    // Demonstrates an insecure practice of inserting sensitive information into debugging logs.

    public static void main(String[] args) {
        String username = "user";
        String password = "password123"; // Sensitive information

        // Debugging code that logs sensitive information
        System.out.println("DEBUG: Attempting login with username: " + username + " and password: " + password);
        
        // Placeholder for login logic
        performLogin(username, password);
    }

    private static void performLogin(String username, String password) {
        // Placeholder for actual login logic
    }
}
