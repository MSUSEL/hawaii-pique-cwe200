import java.net.URL;

public class BAD_URLSensitiveInfo {
    public void createURLWithSensitiveInfo(String token) {
        try {
            URL myUrl = new URL("http://example.com/api?token=" + token); 
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
