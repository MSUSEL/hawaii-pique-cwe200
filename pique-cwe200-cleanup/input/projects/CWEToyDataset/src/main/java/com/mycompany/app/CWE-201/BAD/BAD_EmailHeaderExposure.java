import javax.mail.*;
import javax.mail.internet.*;
import java.util.Properties;

public class BAD_EmailHeaderExposure {
    public void sendEmailWithSensitiveHeader(String recipient, String sessionToken) {
        Properties properties = System.getProperties();
        properties.setProperty("mail.smtp.host", "smtp.internal.books.com");
        Session session = Session.getDefaultInstance(properties, null);

        try {
            MimeMessage message = new MimeMessage(session);
            message.setRecipient(Message.RecipientType.TO, new InternetAddress(recipient));
            message.setSubject("Here is your session Info");
            message.addHeader("X-Session-Token", sessionToken);
            Transport.send(message);
        } catch (MessagingException e) {
            System.out.println("Failed to send email: " + e.getMessage());
        }
    }
}
