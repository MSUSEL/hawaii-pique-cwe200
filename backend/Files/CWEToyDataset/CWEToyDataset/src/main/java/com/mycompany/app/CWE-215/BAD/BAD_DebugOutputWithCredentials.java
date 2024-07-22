import java.util.HashMap;
import java.util.Map;

public class BAD_DebugOutputWithCredentials {
    private static final boolean DEBUG_ENABLED = true; // Debugging flag, should be false in production
    private static final Map<String, String> userStore = new HashMap<>();

    public static boolean authenticateUser(String username, String password) {
        // Simulated authentication logic
        boolean isAuthenticated = userStore.containsKey(username) && userStore.get(username).equals(password);

        if (DEBUG_ENABLED) {
            if (isAuthenticated) {
                System.out.println("DEBUG: Authentication succeeded for user: " + username);
            } else {
                System.out.println("DEBUG: Authentication failed for user: " + username + " with password: " + password);
            }
        }

        return isAuthenticated;
    }

    public static void main(String[] args) {
        authenticateUser("admin", "wrongPassword");

        authenticateUser("admin", "adminSecret");
    }
}
