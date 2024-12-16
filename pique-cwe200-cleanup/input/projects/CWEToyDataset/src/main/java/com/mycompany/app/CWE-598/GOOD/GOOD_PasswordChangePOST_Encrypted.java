import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class GOOD_PasswordChangePOST_Encrypted {
    private static SecretKey accountKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // Using AES with 128-bit keys.
            accountKey = keyGen.generateKey();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static String encrypt(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, accountKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void changePassword(String username, String oldPassword, String newPassword) {
        HttpURLConnection connection = null;
        try {
            URL url = new URL("https://secure.companyportal.com/changePassword");
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true); // Allows sending data
            connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

            // Encrypt and encode parameters
            username = URLEncoder.encode(encrypt(username), StandardCharsets.UTF_8.name());
            oldPassword = URLEncoder.encode(encrypt(oldPassword), StandardCharsets.UTF_8.name());
            newPassword = URLEncoder.encode(encrypt(newPassword), StandardCharsets.UTF_8.name());

            // Construct POST data
            String postData = "username=" + username + "&oldPassword=" + oldPassword + "&newPassword=" + newPassword;

            try (OutputStream os = connection.getOutputStream()) {
                os.write(postData.getBytes());
                os.flush();
            }

        } catch (Exception e) {
            System.err.println("Error occurred while changing password");
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    
    public static void main(String[] args) {
        changePassword("user2", "DragonHeart21", "ShadowHunter34"); 
    }
}
