import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;
import java.util.UUID;

public class BAD_LogPaymentInformation {
    private static final Logger logger = LogManager.getLogger(BAD_LogPaymentInformation.class);

    public boolean validatePaymentDetails(String creditCardNumber, String expiryDate, String cvv) {
        logger.info("Processing payment with card number: " + creditCardNumber + ", expiry date: " + expiryDate + ", and CVV: " + cvv);

        return creditCardNumber.length() == 16 && cvv.length() == 3; // Simplified validation
    }

    public String generateTransactionId() {
        // Generate a unique transaction ID for tracking
        return UUID.randomUUID().toString();
    }

    public void processPayment(String creditCardNumber, String expiryDate, String cvv) {
        String transactionId = generateTransactionId();

        if (!validatePaymentDetails(creditCardNumber, expiryDate, cvv)) {
            logger.error("Invalid payment details provided for transaction " + creditCardNumber + ", expiry date: " + expiryDate + ", and CVV: " + cvv);
            return; // Early return if validation fails
        }

        // Simulate payment processing logic...
        boolean paymentSuccess = true; // Placeholder for actual payment processing result
    }

    public static void main(String[] args) {
        new BAD_LogPaymentInformation().processPayment("1234567890123456", "12/23", "123");
    }
}
