import org.apache.http.client.fluent.Request;
import org.apache.http.entity.ContentType;
import java.util.Properties;
import java.io.FileInputStream;
import java.nio.file.Paths;
/*
 * This exposes sensative data in the query parameters (CWE-598 falls under this as well). 
 * However I made it a bit harder to track then "BAD_SendSensitiveInfoInGetRequest.java" by using different methods, and variables along with concatenation. 
 */

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
            System.out.println("There was an error processing your credit card: " + creditCardNumber);
        }
    }

    private static String getCreditCardNumberFromSecureSource() {        
        try {
            Properties prop = new Properties();
            prop.load(new FileInputStream(Paths.get("config.properties").toFile()));
            return prop.getProperty("creditCardNumber");
        } catch (Exception e) {
            System.err.println("An error occured reading config file.");
            return null;
        }
    }


    public static void main(String[] args) {
        String CCNum = getCreditCardNumberFromSecureSource();
        sendCreditCardInfo(CCNum);
    }
}
