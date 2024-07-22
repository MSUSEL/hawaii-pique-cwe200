public class GOOD_DebugLoggingSensitiveInfo {
    private static final boolean DEBUG_FLAG = System.getProperty("debugMode", "false").equalsIgnoreCase("true"); // Debugging flag, should be false in production

    public static void main(String[] args) {
        String username = System.getProperty("username", "defaultUser");
        String password = System.getProperty("password", "defaultPass");

        if (DEBUG_FLAG) {
            System.out.println("DEBUG: Attempting login.");
        }

        // Placeholder for login logic
        performLogin(username, password);
    }

    private static void performLogin(String username, String password) {
        // Placeholder for actual login logic
    }
}
