import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_FetchPublicData {
    
    public static void submitSportsCategory(String category) {
        try {
            String urlString = "http://sports.com/stats?sport=" + category;
                               
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        submitSportsCategory("Football");
    }
}
