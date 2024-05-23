import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.Base64;

public class GOOD_ApiKeySubmissionGET {
    private static SecretKey secretKey;

    static {
        try {
            // Generate a key for encryption
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // AES 128-bit key
            secretKey = keyGen.generateKey();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String encryptApiKey(String apiKey) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(apiKey.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void submitApiKey(String apiKey) {
        try {
            apiKey = encryptApiKey(apiKey);
            URI uri = new URI("http", "example.com", "/api/submit", "apiKey=" + apiKey, null);
            URL url = uri.toURL();
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");

            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
            submitApiKey("secretApiKey123"); // Send encrypted API key
        } catch (Exception e) {
            System.out.println("Error submitting API key: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
