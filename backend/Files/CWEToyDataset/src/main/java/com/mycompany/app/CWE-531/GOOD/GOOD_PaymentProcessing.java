import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

public class GOOD_PaymentProcessing {
    private Properties testProps;

    @BeforeEach
    void setUp() throws IOException {
        testProps = new Properties();
        try (FileInputStream in = new FileInputStream("test_payment_config.properties")) {
            testProps.load(in);
        }
    }

    @Test
    void testProcessPayment() {
        PaymentProcessor processor = new PaymentProcessor();
        String creditCardNumber = testProps.getProperty("creditCardNumber");
        double amount = Double.parseDouble(testProps.getProperty("amount"));
        assertTrue(processor.processPayment(creditCardNumber, amount), "Payment should be processed");
    }
}
