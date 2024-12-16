import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_FinancialDataGET {
    public static void sendFinancialData(String accountNumber, String amount) {
        try {
            String urlString = "http://payments.fintechservice.com/processPayment?account=" + accountNumber + "&amount=" + amount;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        sendFinancialData("123456789", "100.00");
    }
}
