import java.net.URL;
import java.net.URLConnection;
import java.net.HttpURLConnection;

public class BAD_SessionTokenGET {
    public static void retrieveData(String sessionToken) {
        try {
            String urlString = "http://api.internalnetwork.com/data?sessionToken=" + sessionToken;
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }
    
    public static void main(String[] args) {
        retrieveData("abc123sessionToken");
    }
}
