import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.assertTrue;
import java.util.Properties;

class EmailNotificationService {
    public boolean sendEmail(String recipientEmail, String message) {
        // This would interface with an email server in production
        return recipientEmail.contains("@") && !message.isEmpty();
    }
}

public class GOOD_EmailNotificationTest {
    private Properties properties;
    private EmailNotificationService emailService;

    @BeforeEach
    public void setup() {
        // Load configuration properties
        properties = new Properties();
        properties.setProperty("recipientEmail", System.getenv("TEST_RECIPIENT_EMAIL"));
        properties.setProperty("message", "This is a test notification.");

        // Initialize the service
        emailService = new EmailNotificationService();
    }

    @Test
    public void testSendEmail() {
        String recipientEmail = properties.getProperty("recipientEmail");
        String message = properties.getProperty("message");

        // Validate that the recipient email and message are set
        if (recipientEmail == null || message == null) {
            throw new IllegalStateException("Test configuration must include recipient email and message.");
        }

        // Perform the test
        boolean result = emailService.sendEmail(recipientEmail, message);
        assertTrue(result, "Email should be sent successfully.");
    }
}
