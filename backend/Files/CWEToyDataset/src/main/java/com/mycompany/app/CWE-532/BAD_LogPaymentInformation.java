import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

public class BAD_LogPaymentInformation {
    private static final Logger logger = LogManager.getLogger(BAD_LogPaymentInformation.class);

    public void processPayment(String creditCardNumber, String expiryDate, String cvv) {
        // Insecurely logging full payment information
        logger.info("Processing payment with card number: " + creditCardNumber + ", expiry date: " + expiryDate + ", and CVV: " + cvv);

        // Payment processing logic...
    }

    public static void main(String[] args) {
        new BAD_LogPaymentInformation().processPayment("1234567890123456", "12/23", "123");
    }
}
