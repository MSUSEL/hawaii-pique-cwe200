import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;

public class BAD_ApiKeySubmissionGET {
    public static void submitApiKey(String apiKey) {
        try {
            URI uri = new URI("http", "corporateportal.companydomain.com", "/api/submit", "apiKey=" + apiKey, null);
            URL url = uri.toURL();
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }
    
    public static void main(String[] args) {
        submitApiKey("abc12345-def67890-ghij1234-klmn5678");
    }
}
