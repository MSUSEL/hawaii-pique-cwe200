import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_CreditCardSubmissionGET {
    
    public static void submitCreditCard(String cardNumber, String expiryDate, String cvv) {
        try {
            String urlString = "http://example.com/submitPayment?cardNumber=" + cardNumber +
                               "&expiryDate=" + expiryDate + "&cvv=" + cvv; // Highly sensitive data in query
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
        submitCreditCard("1234567890123456", "12/23", "123"); // Extremely risky!
    }
}
