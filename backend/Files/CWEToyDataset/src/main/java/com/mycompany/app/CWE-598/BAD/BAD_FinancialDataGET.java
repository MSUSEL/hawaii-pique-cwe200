import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_FinancialDataGET {
    public static void sendFinancialData(String accountNumber, String amount) {
        try {
            String urlString = "http://example.com/processPayment?account=" + accountNumber + "&amount=" + amount; // Sensitive data in query
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending financial data via 'GET' request to URL: " + url);
            System.out.println("Response Code: " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        sendFinancialData("123456789", "100.00"); // Extremely risky!
    }
}
