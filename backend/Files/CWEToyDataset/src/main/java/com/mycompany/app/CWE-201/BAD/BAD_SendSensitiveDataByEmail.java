import java.util.Properties;
import javax.mail.*;
import javax.mail.internet.*;

public class BAD_SendSensitiveDataByEmail {
    public void sendEmail(String recipient, String sensitiveData) throws MessagingException {
        String host = "smtp.example.com";
        Properties properties = System.getProperties();
        properties.setProperty("mail.smtp.host", host);
        Session session = Session.getDefaultInstance(properties);

        try {
            MimeMessage message = new MimeMessage(session);
            message.setFrom(new InternetAddress("noreply@example.com"));
            message.addRecipient(Message.RecipientType.TO, new InternetAddress(recipient));
            // BAD: Sending sensitive data in email content
            message.setSubject("Your Sensitive Data");
            message.setText("Here is your sensitive data: " + sensitiveData);
            Transport.send(message);
        } catch (MessagingException mex) {
            mex.printStackTrace();
        }
    }
}