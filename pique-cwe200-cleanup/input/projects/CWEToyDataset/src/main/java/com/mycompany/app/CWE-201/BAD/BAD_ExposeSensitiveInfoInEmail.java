import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.util.Properties;

public class BAD_ExposeSensitiveInfoInEmail {
    public void sendEmail(String recipient, String creditCardNumber) {
        Properties properties = System.getProperties();
        properties.setProperty("mail.smtp.host", "walmart.internal.com");
        Session session = Session.getDefaultInstance(properties);

        try {
            MimeMessage message = new MimeMessage(session);
            message.setFrom(new InternetAddress("fredsemailm@walmart.com"));
            message.addRecipient(Message.RecipientType.TO, new InternetAddress(recipient));
            message.setSubject("Fred here's your personal credit card number");
            message.setText("Your credit card number is: " + creditCardNumber + "and the CVV is 987");
            Transport.send(message);
        } catch (MessagingException mex) {
            mex.printStackTrace();
        }
    }
}
