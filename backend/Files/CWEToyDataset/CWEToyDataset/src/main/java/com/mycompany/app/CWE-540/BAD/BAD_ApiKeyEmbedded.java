import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_ApiKeyEmbedded {
    // Hardcoded API key for external service
    private static final String SERVICE_API_KEY = "ABCD1234EFGH5678";

    public static void callExternalService() {
        try {
            URL url = new URL("https://api.externalservice.com/data?api_key=" + SERVICE_API_KEY);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            // Check the HTTP response code, etc...
            System.out.println("Response Code: " + conn.getResponseCode());
            // Further processing...
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        callExternalService();
    }
}