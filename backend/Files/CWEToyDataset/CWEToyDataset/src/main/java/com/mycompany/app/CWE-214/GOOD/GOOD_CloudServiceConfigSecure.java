import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_CloudServiceConfigSecure {
    public static void main(String[] args) {
        try {
            String cloudAccessToken = "cloudAccessSecret!";
            String encryptedToken = encryptToken(cloudAccessToken);

            ProcessBuilder builder = new ProcessBuilder("cloud", "deploy", "--access-token", encryptedToken, "my-cloud-app");
            Process process = builder.start();
            process.waitFor();  // Wait for the process to complete
            System.out.println("Cloud deployment initiated successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Encrypt the token using AES encryption
    private static String encryptToken(String token) throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(128);  // Use AES-128 encryption
        SecretKey secretKey = keyGen.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encryptedBytes = cipher.doFinal(token.getBytes());

        // Encode bytes to Base64 to ensure safe transmission over command line
        return Base64.getEncoder().encodeToString(encryptedBytes);
    }
}
