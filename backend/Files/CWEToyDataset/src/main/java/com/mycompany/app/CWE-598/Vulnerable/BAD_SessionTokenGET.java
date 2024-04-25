import java.net.URL;
import java.net.URLConnection;
import java.net.HttpURLConnection;

public class BAD_SessionTokenGET {
    public static void retrieveData(String sessionToken) {
        try {
            String urlString = "http://example.com/data?sessionToken=" + sessionToken; // Sensitive session token in query
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            
            System.out.println("Accessing data with 'GET' request to URL : " + url);
            System.out.println("Response Code : " + ((HttpURLConnection) connection).getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        retrieveData("abc123sessionToken"); // Extremely risky!
    }
}
