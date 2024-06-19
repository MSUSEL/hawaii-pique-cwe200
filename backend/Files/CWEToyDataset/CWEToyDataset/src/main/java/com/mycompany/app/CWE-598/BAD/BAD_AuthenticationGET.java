import java.net.URL;
import java.net.URLConnection;
import java.io.InputStream;

public class BAD_AuthenticationGET {
    public static void authenticateUser(String userId, String authToken) {
        try {
            String urlString = "http://example.com/auth?userId=" + userId + "&token=" + authToken;
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            InputStream responseStream = connection.getInputStream();
            
            responseStream.close();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        authenticateUser("user123", "secureTokenXYZ");
    }
}
