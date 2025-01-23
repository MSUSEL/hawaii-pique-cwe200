import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.util.Base64;
import java.net.URLEncoder;

public class GOOD_CreditCardSubmissionPOST_Encrypted {

    private static SecretKey paymentKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // Using AES with 128-bit keys.
            paymentKey = keyGen.generateKey();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static String encrypt(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, paymentKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void submitCreditCard(String cardNumber, String expiryDate, String cvv) {
        try {
            cardNumber = encrypt(cardNumber);
            expiryDate = encrypt(expiryDate);
            cvv = encrypt(cvv);

            URL url = new URL("https://bankingAmerica.com/submitPayment");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);  // Necessary for POST
            connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

            // Constructing the data to be sent
            String urlParameters = "cardNumber=" + URLEncoder.encode(cardNumber, "UTF-8") +
                                   "&expiryDate=" + URLEncoder.encode(expiryDate, "UTF-8") +
                                   "&cvv=" + URLEncoder.encode(cvv, "UTF-8");

            try (OutputStream os = connection.getOutputStream()) {
                os.write(urlParameters.getBytes());
                os.flush();
            }

        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        submitCreditCard("1234567890123456", "12/23", "123");
    }
}
