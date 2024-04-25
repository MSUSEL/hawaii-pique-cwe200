import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmailService {
    public boolean sendEmail(String email, String message) {
        // Pretend to send an email
        return email.contains("@") && !message.isEmpty();
    }
}

public class BAD_EmailService {
    @Test
    public void testSendEmail() {
        String recipientEmail = "user@example.com";
        String message = "Your password is password123!"; // Sensitive information in test
        EmailService emailService = new EmailService();
        assertTrue(emailService.sendEmail(recipientEmail, message));
    }
}
