import javax.swing.JOptionPane;

public class GOOD_ExplicitUserConsent {
    // This class demonstrates obtaining explicit user consent before sharing data with analytics services.
    // Good Practice: Recognizes that user data is sensitive across different policies and ensures consent is obtained in accordance with user expectations.

    public boolean obtainUserConsent() {
        int consent = JOptionPane.showConfirmDialog(null, "Do you agree to share your data with our analytics partners?", "User Consent", JOptionPane.YES_NO_OPTION);
        return consent == JOptionPane.YES_OPTION;
    }

    public void sendDataToAnalytics(String userData) {
        if (obtainUserConsent()) {
            System.out.println("User agreed to data sharing. Sending data to analytics.");
            // Code to send data to analytics...
        } else {
            System.out.println("User did not agree to data sharing. Data will not be sent.");
        }
    }

    public static void main(String[] args) {
        new GOOD_ExplicitUserConsent().sendDataToAnalytics("User's private browsing data");
    }
}
