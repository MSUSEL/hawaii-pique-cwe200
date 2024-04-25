import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;

public class BAD_ApiKeySubmissionGET {
    public static void submitApiKey(String apiKey) {
        try {
            URI uri = new URI("http", "example.com", "/api/submit", "apiKey=" + apiKey, null);
            URL url = uri.toURL();
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        submitApiKey("secretApiKey123"); // Extremely risky!
    }
}
