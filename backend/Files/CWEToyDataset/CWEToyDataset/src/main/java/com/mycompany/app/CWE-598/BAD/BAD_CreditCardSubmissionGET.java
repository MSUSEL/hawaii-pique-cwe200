import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_CreditCardSubmissionGET {
    
    public static void submitCreditCard(String cardNumber, String expiryDate, String cvv) {
        try {
            String urlString = "http://example.com/submitPayment?cardNumber=" + cardNumber +
                               "&expiryDate=" + expiryDate + "&cvv=" + cvv; 
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }
    
    public static void main(String[] args) {
        submitCreditCard("1234567890123456", "12/23", "123");
    }
}
