import java.net.URL;
import java.net.URLConnection;
import java.io.InputStream;

public class BAD_AuthenticationGET {
    public static void authenticateUser(String userId, String authToken) {
        try {
            String urlString = "http://example.com/auth?userId=" + userId + "&token=" + authToken; // Sensitive data in query
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            
            InputStream responseStream = connection.getInputStream();
            System.out.println("Authenticating with 'GET' request to URL: " + url);

            // Handling the response (omitted for brevity)
            responseStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        authenticateUser("user123", "secureTokenXYZ"); // Extremely risky!
    }
}
