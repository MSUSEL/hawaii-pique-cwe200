import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;
import java.nio.charset.StandardCharsets;

public class GOOD_PasswordChangeGET_Encrypted {
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

    public static void changePassword(String username, String oldPassword, String newPassword) {
        try {
            username = URLEncoder.encode(encrypt(username), StandardCharsets.UTF_8.name());
            oldPassword = URLEncoder.encode(encrypt(oldPassword), StandardCharsets.UTF_8.name());
            newPassword = URLEncoder.encode(encrypt(newPassword), StandardCharsets.UTF_8.name());

            String urlString = "https://example.com/changePassword?username=" + username + 
                               "&oldPassword=" + oldPassword + "&newPassword=" + newPassword; // Encrypted data in query
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
        changePassword("user2", "oldPass123", "newPass456"); // Now with encrypted parameters
    }
}
