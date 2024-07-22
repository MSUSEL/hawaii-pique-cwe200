import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_UseHttpsForSensitiveData {
    public static void transmitDataSecurely(String data) {
        try {
            URL url = new URL("https://internalapi.com/secure-api");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");
            
            connection.getOutputStream().write(data.getBytes());
            System.out.println("Securely transmitted data. Response Code: " + connection.getResponseCode());
        } catch (Exception e) {
            System.err.println("An error has occurred.");
        }
    }

    public static void main(String[] args) {
        String sensitiveData = "{\"message\": \"This is sensitive\"}";
        transmitDataSecurely(sensitiveData);
    }
}
