import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_FetchWeatherData {
    
    public static void submitSportsCategory(String city) {
        try {
            String urlString = "http://weather.com/getWeather?city=" + city;
                               
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            System.err.println("Error occurred while fetching weather data.");
        }
    }
    
    public static void main(String[] args) {
        submitSportsCategory("Honolulu");
    }
}
