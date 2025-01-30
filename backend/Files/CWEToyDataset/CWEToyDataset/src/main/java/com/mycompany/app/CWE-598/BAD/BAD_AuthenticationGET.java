import java.net.URL;
import java.net.URLConnection;
import java.io.InputStream;
import java.net.HttpURLConnection;


public class BAD_AuthenticationGET {
    public static void authenticateUser(String userId, String authToken) {
        try {
            String urlString = "http://auth.companyportal.com/auth?userId=" + userId + "&token=" + authToken;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        authenticateUser("user123", "secureTokenXYZ");
    }
}
