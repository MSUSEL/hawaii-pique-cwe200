public class GOOD_DebugLoggingSensitiveInfo {
    // Demonstrates an insecure practice of inserting sensitive information into debugging logs.
    private static final boolean DEBUG_FLAG = System.getProperty("debugMode", "false").equalsIgnoreCase("true"); // Debugging flag, should be false in production

    public static void main(String[] args) {
        // Fetching username and password from system properties or defaulting to safe values
        String username = System.getProperty("username", "defaultUser");
        String password = System.getProperty("password", "defaultPass"); // Sensitive information

        if (DEBUG_FLAG) {
            // Debugging code that logs sensitive information
            System.out.println("DEBUG: Attempting login.");
        }

        // Placeholder for login logic
        performLogin(username, password);
    }

    private static void performLogin(String username, String password) {
        // Placeholder for actual login logic
    }
}
