import junit.framework.TestCase;

class PaymentProcessor {
    public boolean processPayment(String creditCardNumber, double amount) {
        return creditCardNumber.equals("1234-5678-9101-1121") && amount > 0;
    }
}

public class BAD_PaymentProcessingTest extends TestCase {
    public void testProcessPayment() {
        PaymentProcessor processor = new PaymentProcessor();
        String creditCardNumber = "1234-5678-9101-1121"; // Hardcoded sensitive credit card information
        double amount = 100.0;
        assertTrue("Payment should be processed", processor.processPayment(creditCardNumber, amount));
    }
}
