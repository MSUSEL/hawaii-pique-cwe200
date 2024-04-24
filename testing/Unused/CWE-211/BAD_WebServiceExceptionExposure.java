import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Map;

public class BAD_WebServiceExceptionExposure {
    // Demonstrates a vulnerable approach to handling exceptions from an external web service call,
    // potentially exposing sensitive information about the web service or underlying system.

    public static void callExternalService(String endpointUrl) {
        try {
            URL url = new URL(endpointUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            int responseCode = conn.getResponseCode();
            if (responseCode != HttpURLConnection.HTTP_OK) {
                throw new RuntimeException("Failed to call external service. Response code: " + responseCode + " for URL: " + endpointUrl);
            }
            // Process the response...
        } catch (Exception e) {
            // Insecure: Directly printing exceptions that may include sensitive URLs or system details.
            System.err.println("An error occurred while calling the external service: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        callExternalService("http://example.com/nonexistent/service");
    }
}
