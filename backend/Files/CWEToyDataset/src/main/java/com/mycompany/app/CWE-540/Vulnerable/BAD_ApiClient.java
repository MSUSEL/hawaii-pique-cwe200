import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_ApiClient {
    private static final String API_KEY = "ABCD1234APIKEYSECRET";  // Sensitive API key included in source code

    public void sendData() throws Exception {
        URL url = new URL("https://api.example.com/data");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestProperty("Authorization", "Bearer " + API_KEY);
        conn.setRequestMethod("GET");
        // Process the response
    }
}
