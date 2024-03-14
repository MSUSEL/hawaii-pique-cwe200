import javax.mail.*;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.util.Properties;

// This example demostrates sending sensitive information via email. 
public class BAD_SendSensitiveDataViaEmail {
    public static void sendEmail(String recipient, String sensitiveContent) {
        // Fetching email credentials and server details from environment variables
        final String username = System.getenv("EMAIL_USERNAME");
        final String password = System.getenv("EMAIL_PASSWORD");
        final String smtpHost = System.getenv("SMTP_HOST");
        final String smtpPort = System.getenv("SMTP_PORT");

        Properties prop = new Properties();
        prop.put("mail.smtp.host", smtpHost);
        prop.put("mail.smtp.port", smtpPort);
        prop.put("mail.smtp.auth", "true");
        prop.put("mail.smtp.starttls.enable", "true");

        Session session = Session.getInstance(prop, new Authenticator() {
            protected PasswordAuthentication getPasswordAuthentication() {
                return new PasswordAuthentication(username, password);
            }
        });

        try {
            Message message = new MimeMessage(session);
            message.setFrom(new InternetAddress(username)); // Using the sender's email from environment
            message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(recipient));
            message.setSubject("Sensitive Information");

            // It's crucial to ensure that the sensitive content itself is also securely handled
            message.setText("Here is some sensitive information: " + sensitiveContent);

            Transport.send(message);

            System.out.println("Sent sensitive information via email successfully.");
        } catch (MessagingException e) {
            System.err.println("An error has occured sending an email.");
        }
    }

    public static void main(String[] args) {
        // Example usage - ensure that sensitiveContent is securely fetched or handled
        sendEmail("recipient@example.com", "Sensitive content from a secure source.");
    }
}
