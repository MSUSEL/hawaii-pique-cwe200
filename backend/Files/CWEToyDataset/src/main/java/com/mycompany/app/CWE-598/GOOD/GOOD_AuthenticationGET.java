import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.net.URL;
import java.net.URLConnection;
import java.io.InputStream;
import java.util.Base64;

public class GOOD_AuthenticationGET {
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

    public static String encryptCredentials(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void authenticateUser(String userId, String authToken) {
        try {
            userId = encryptCredentials(userId);
            authToken = encryptCredentials(authToken);
            String urlString = "http://example.com/auth?userId=" + userId + "&token=" + authToken; // Encrypted data in query
            URL url = new URL(urlString);
            URLConnection connection = url.openConnection();
            
            InputStream responseStream = connection.getInputStream();
            System.out.println("Authenticating with 'GET' request to URL: " + url);

            // Handling the response (omitted for brevity)
            responseStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        authenticateUser("user123", "secureTokenXYZ"); // Now with encrypted parameters
    }
}
