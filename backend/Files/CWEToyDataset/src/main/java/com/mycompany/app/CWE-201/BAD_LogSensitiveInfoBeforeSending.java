import org.apache.http.client.fluent.Request;
import org.apache.http.entity.ContentType;

public class BAD_LogSensitiveInfoBeforeSending {
    public static void sendCreditCardInfo(String creditCardNumber) {
        String payload = "{\"creditCardNumber\": \"" + creditCardNumber + "\"}";
        
        // Logging sensitive information
        System.out.println("Sending payload: " + payload);
        
        try {
            String response = Request.Post("https://example.com/api/payment")
                    .bodyString(payload, ContentType.APPLICATION_JSON)
                    .execute().returnContent().asString();
            // Further processing
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        sendCreditCardInfo("1234-5678-9012-3456");
    }
}
