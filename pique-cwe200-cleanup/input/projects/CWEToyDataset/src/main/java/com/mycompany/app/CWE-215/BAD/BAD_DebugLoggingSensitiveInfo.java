public class BAD_DebugLoggingSensitiveInfo {
    private static final boolean DEBUG_FLAG = System.getProperty("debugMode", "false").equalsIgnoreCase("true"); // Debugging flag, should be false in production

    public static void main(String[] args) {
        // Fetching username and password from system properties or defaulting to safe values
        String username = System.getProperty("username", "defaultUser");
        String password = System.getProperty("password", "defaultPass");

        if (DEBUG_FLAG) {
            System.out.println("DEBUG: Attempting login with username: " + username + " and password: " + password);
        }

        // Placeholder for login logic
        performLogin(username, password);
    }

    private static void performLogin(String username, String password) {
        // Placeholder for actual login logic
    }
}
