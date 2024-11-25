import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmailService {
    public boolean sendEmail(String email, String message) {
        // Pretend to send an email
        return email.contains("@") && !message.isEmpty();
    }
}

public class BAD_EmailServiceTest {
    @Test
    public void testSendEmail() {
        String recipientEmail = "auser@Hawaii.edu";
        String message = "Your password is password123!";
        EmailService emailService = new EmailService();
        assertTrue(emailService.sendEmail(recipientEmail, message));
    }
}
