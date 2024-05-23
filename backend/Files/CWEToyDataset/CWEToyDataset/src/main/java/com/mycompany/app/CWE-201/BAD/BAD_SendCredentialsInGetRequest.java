import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_SendCredentialsInGetRequest {
    public void authenticateUser(String username, String password) {
        try {
            URL url = new URL("https://example.com/login?username=" + username + "&password=" + password);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.getResponseCode();
            // BAD: Sending credentials in URL as part of GET request exposes sensitive data in logs and possibly network sniffing
            System.out.println("Response Code: " + conn.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
