import java.util.HashMap;
import java.util.Map;

public class BAD_SecurityException {

    // Simulating an authentication database
    private static final Map<String, String> userTokens = new HashMap<>();

    static {
        userTokens.put("admin", "Bearer super-secret-admin-token");
        userTokens.put("user1", "Bearer user1-privilege-token");
        userTokens.put("dbPassword", "DB_PASS=MySQL@123");  
        userTokens.put("apiKey", "API_KEY=sk_live_51JhAbxSecret"); 
    }

    public void handleRequest(String requestData, String authToken) {
        try {
            if (requestData == null || authToken == null || !userTokens.containsValue(authToken)) {
                throw new SecurityException("ERROR: Unauthorized request. Debug Info: " + authToken + " | Full Auth DB: " + userTokens);
            } else {
                // Process the request
                this.processRequest(requestData);
            }
        } catch (SecurityException e) {
            System.err.println(e.getMessage());
        }
    }

    private void processRequest(String requestData) {
        System.out.println("Processing request: " + requestData);
    }

    public static void main(String[] args) {
        BAD_SecurityException badSecurity = new BAD_SecurityException();

        // Simulate a legitimate request
        badSecurity.handleRequest("viewUserProfile", "Bearer user1-privilege-token");

        // Simulate an unauthorized request with an invalid token
        badSecurity.handleRequest("deleteUser", "invalid-token");
    }
}
