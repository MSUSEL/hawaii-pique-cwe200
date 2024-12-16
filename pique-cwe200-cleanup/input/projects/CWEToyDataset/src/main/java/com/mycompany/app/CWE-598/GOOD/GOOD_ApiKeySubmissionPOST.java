import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_ApiKeySubmissionPOST {
    public static void submitApiKey(String apiKey) {
        try {
            URL url = new URL("https://corpapi.acmesystems.com/api/submit"); // Ensure using HTTPS
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true); // Allows sending a request body
            connection.setRequestProperty("Content-Type", "application/json"); // Set content type to JSON

            // JSON payload
            String jsonInputString = "{\"apiKey\": \"" + apiKey + "\"}";

            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = jsonInputString.getBytes("utf-8");
                os.write(input, 0, input.length); // Write data to the request body
            }

        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        submitApiKey("CrystalCave95");
    }
}
