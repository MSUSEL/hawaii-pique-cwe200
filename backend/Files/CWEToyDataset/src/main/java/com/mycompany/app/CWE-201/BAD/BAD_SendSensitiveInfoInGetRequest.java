import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;

// This exposes sensative data in the query parameters (CWE-598 falls under this as well).
public class BAD_SendSensitiveInfoInGetRequest {
    public static void sendSensitiveData(String username, String password) {
        try {
            URL url = new URL("http://example.com/login?username=" + username + "&password=" + password);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            // Processing the response...
            System.out.println("Response Code: " + connection.getResponseCode());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        final String username = System.getenv("EMAIL_USERNAME");
        final String password = System.getenv("EMAIL_PASSWORD");
        sendSensitiveData(username, password);
    }
}
