import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Base64;

public class GOOD_PasswordSecureCommands {
    public static void main(String[] args) {
        try {
            String authToken = getEncryptedAuthToken();  // Retrieve securely stored token
            
            URL url = new URL("http://example.com/data");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Authorization", "Basic " + authToken);
            conn.connect();

            int responseCode = conn.getResponseCode();
            System.out.println("Response Code: " + responseCode);
            // Handle the response appropriately

            conn.disconnect();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String getEncryptedAuthToken() {
        // Simulate retrieving an encrypted token and decrypting it
        String username = "admin";
        String password = "mySecretPass123"; // Normally, you would retrieve this securely from a vault
        String authString = username + ":" + password;
        byte[] authEncBytes = Base64.getEncoder().encode(authString.getBytes());
        return new String(authEncBytes);
    }
}