import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_TokenSecureViaEnvironment {
    public static void main(String[] args) {
        try {
            String secretToken = "s3cr3tT0k3n";
            String encryptedToken = encryptToken(secretToken);

            ProcessBuilder builder = new ProcessBuilder("myCommand");
            builder.environment().put("SECRET_TOKEN", encryptedToken); // Pass encrypted token
            Process process = builder.start();
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String encryptToken(String token) throws Exception {
        // Encrypt the token using AES encryption
        KeyGenerator keyGenerator = KeyGenerator.getInstance("AES");
        keyGenerator.init(128); // Use AES-128 encryption
        SecretKey secretKey = keyGenerator.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encryptedBytes = cipher.doFinal(token.getBytes());

        return Base64.getEncoder().encodeToString(encryptedBytes);
    }
}
