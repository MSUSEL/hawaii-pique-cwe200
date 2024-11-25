import java.util.Properties;
import javax.mail.*;
import javax.mail.internet.*;

public class BAD_SendSensitiveDataByEmail {
    public void sendEmail(String recipient, String verificationCode) throws MessagingException {
        String host = "uh.mailservice.edu";
        Properties properties = System.getProperties();
        properties.setProperty("mail.smtp.host", host);
        Session session = Session.getDefaultInstance(properties);

        try {
            MimeMessage message = new MimeMessage(session);
            message.setFrom(new InternetAddress("cwe200@uh.edu"));
            message.addRecipient(Message.RecipientType.TO, new InternetAddress(recipient));
            message.setSubject("Verification code");
            message.setText("Here is your verification code: " + verificationCode);
            Transport.send(message);
        } catch (MessagingException mex) {
            mex.printStackTrace();
        }
    }
}
