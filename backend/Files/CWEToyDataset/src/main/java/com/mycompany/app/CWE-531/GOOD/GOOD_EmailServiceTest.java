import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;


public class GOOD_EmailServiceTest {
    @Test
    public void testSendEmail() {
        // Fetch the recipient email from environment variables
        String recipientEmail = System.getenv("RECIPIENT_EMAIL");
     
        String message = "Hello, your request has been processed successfully."; // Generic, non-sensitive information
        EmailService emailService = new EmailService();
        assertTrue(emailService.sendEmail(recipientEmail, message));
    }

    class EmailService {
        public boolean sendEmail(String email, String message) {
            // Pretend to send an email
            return email.contains("@") && !message.isEmpty();
        }
    }
}
