import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_FetchPublicData {
    
    public static void submitSportsCategory(String category) {
        try {
            String urlString = "http://sports.com/stats?sport=" + category;
                               
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
 
        } catch (Exception e) {
            System.err.println("Error occurred while fetching sports data.");
        }
    }
    
    public static void main(String[] args) {
        submitSportsCategory("Football");
    }
}
