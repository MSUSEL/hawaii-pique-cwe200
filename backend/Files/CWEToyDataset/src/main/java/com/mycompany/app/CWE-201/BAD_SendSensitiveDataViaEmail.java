import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.util.Properties;

public class BAD_SendSensitiveDataViaEmail {
    public static void sendEmail(String recipient, String sensitiveContent) {
        final String username = "yourEmail@example.com"; // Use your real email
        final String password = "emailPassword"; // Use your real password

        Properties prop = new Properties();
        prop.put("mail.smtp.host", "smtp.example.com");
        prop.put("mail.smtp.port", "587");
        prop.put("mail.smtp.auth", "true");
        prop.put("mail.smtp.starttls.enable", "true"); // Note: This enables Transport Layer Security (TLS) but does not encrypt the email content itself

        Session session = Session.getInstance(prop, new javax.mail.Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication(username, password);
            }
        });

        try {
            Message message = new MimeMessage(session);
            message.setFrom(new InternetAddress("yourEmail@example.com"));
            message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(recipient));
            message.setSubject("Sensitive Information");
            message.setText("Here is some sensitive information: " + sensitiveContent);

            Transport.send(message);

            System.out.println("Sent sensitive information via email successfully.");
        } catch (MessagingException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        sendEmail("recipient@example.com", "Here's a secret: ...");
    }
}
