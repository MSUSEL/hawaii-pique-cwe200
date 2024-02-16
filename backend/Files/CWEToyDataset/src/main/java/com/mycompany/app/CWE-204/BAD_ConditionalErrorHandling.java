import java.util.HashMap;
import java.util.Map;

public class BAD_ConditionalErrorHandling {
    private static Map<String, String> userData = new HashMap<>();

    static {
        userData.put("existingUser", "userDataIncludingSensitiveInfo");
    }

    public static String fetchUserData(String username) {
        if (userData.containsKey(username)) {
            // Attempt to fetch and return user data
            try {
                // Simulate a data retrieval process that might throw an error
                return retrieveUserData(username);
            } catch (Exception e) {
                // Insecure: Differentiating the error message if user data exists
                return "Error: Failed to retrieve data for existing user.";
            }
        } else {
            // Generic error message for non-existent users
            return "Error: User does not exist.";
        }
    }

    private static String retrieveUserData(String username) throws Exception {
        // Placeholder for an operation that might fail
        throw new Exception("Database connection error");
    }

    public static void main(String[] args) {
        System.out.println(fetchUserData("existingUser"));
        System.out.println(fetchUserData("nonExistingUser"));
    }
}
