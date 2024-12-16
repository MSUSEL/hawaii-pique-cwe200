import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.util.Base64;
import javax.crypto.SecretKey;
import javax.crypto.KeyGenerator;
import java.net.URLEncoder;

public class GOOD_HttpsApiCall {
    public static void main(String[] args) {
        try {
            String token = getDecryptedToken(System.getenv("ENCRYPTED_OAUTH_TOKEN"));
            URL url = new URL("https://api.retailnetwork.com/v2/orders");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true); // Required for POST
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

            String postData = "authToken=" + URLEncoder.encode(token, "UTF-8");

            try (OutputStream os = conn.getOutputStream()) {
                os.write(postData.getBytes());
                os.flush();
            }

            int responseCode = conn.getResponseCode();
            // Handle response appropriately
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    private static String getDecryptedToken(String encryptedToken) {
        try {
            // Mock example for decryption (use a real encryption algorithm)
            byte[] decodedKey = Base64.getDecoder().decode("QWxhZGRpbjpvcGVuIHNlc2FtZQ==");
            SecretKey originalKey = new SecretKeySpec(decodedKey, 0, decodedKey.length, "AES");

            Cipher cipher = Cipher.getInstance("AES");
            cipher.init(Cipher.DECRYPT_MODE, originalKey);
            byte[] original = cipher.doFinal(Base64.getDecoder().decode(encryptedToken));
            return new String(original);
        } catch (Exception e) {
            System.err.println("Error occurred");
            return null;
        }
    }
}
