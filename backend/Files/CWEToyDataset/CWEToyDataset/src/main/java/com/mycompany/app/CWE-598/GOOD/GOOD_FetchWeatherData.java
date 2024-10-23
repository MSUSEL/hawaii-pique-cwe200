import java.net.HttpURLConnection;
import java.net.URL;

public class GOOD_FetchWeatherData {
    
    public static void getWeatherData(String city) {
        try {
            String urlString = "http://weather.com/getWeather?city=" + city;
                               
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred while fetching weather data.");
        }
    }
    
    public static void main(String[] args) {
        getWeatherData("Honolulu");
    }
}
