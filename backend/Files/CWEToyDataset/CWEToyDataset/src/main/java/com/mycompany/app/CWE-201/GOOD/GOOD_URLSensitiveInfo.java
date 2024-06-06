import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

public class GOOD_URLSensitiveInfo {
    public void createURLWithSensitiveInfo(String token) {
        try {
            URL myUrl = new URL("https://example.com/api");
            HttpURLConnection conn = (HttpURLConnection) myUrl.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            String urlParameters = "token=" + token;
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