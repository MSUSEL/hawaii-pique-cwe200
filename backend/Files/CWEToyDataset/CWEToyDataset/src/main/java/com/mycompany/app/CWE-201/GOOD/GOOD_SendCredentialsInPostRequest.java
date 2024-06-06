import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class GOOD_SendCredentialsInPostRequest {
    public void authenticateUser(String username, String password) {
        try {
            URL url = new URL("https://example.com/login");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            String urlParameters = "username=" + username + "&password=" + password;
            byte[] postData = urlParameters.getBytes(StandardCharsets.UTF_8);
            try(OutputStream wr = conn.getOutputStream()) {
                wr.write(postData);
            }
            conn.getResponseCode();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}