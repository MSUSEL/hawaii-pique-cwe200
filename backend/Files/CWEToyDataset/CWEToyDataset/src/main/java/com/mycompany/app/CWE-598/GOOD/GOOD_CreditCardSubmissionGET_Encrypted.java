import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Base64;

public class GOOD_CreditCardSubmissionGET_Encrypted {

    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // Using AES with 128-bit keys.
            secretKey = keyGen.generateKey();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String encrypt(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void submitCreditCard(String cardNumber, String expiryDate, String cvv) {
        try {
            cardNumber = URLEncoder.encode(encrypt(cardNumber), "UTF-8");
            expiryDate = URLEncoder.encode(encrypt(expiryDate), "UTF-8");
            cvv = URLEncoder.encode(encrypt(cvv), "UTF-8");

            String urlString = "https://example.com/submitPayment?cardNumber=" + cardNumber +
                               "&expiryDate=" + expiryDate + "&cvv=" + cvv;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");

            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        submitCreditCard("1234567890123456", "12/23", "123");
    }
}
