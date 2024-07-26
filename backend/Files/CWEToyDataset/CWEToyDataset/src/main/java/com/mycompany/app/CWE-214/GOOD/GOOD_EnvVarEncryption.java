import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_EnvVarEncryption {

    public static void main(String[] args) throws Exception {
        String API_KEY = args[1];
        String encryptedApiKey = encryptApiKey(API_KEY);
        
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("someCommand");
            processBuilder.environment().put("ApiKey", encryptedApiKey);
            processBuilder.start();
            System.out.println("Process invoked with sensitive information in environment variable.");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("An error occurred while invoking the process.");
        }
    }

    private static String encryptApiKey(String apiKey) throws Exception {
        KeyGenerator keyGenerator = KeyGenerator.getInstance("AES");
        keyGenerator.init(128); // Use AES-128 for encryption
        SecretKey authKey = keyGenerator.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, authKey);
        byte[] encryptedBytes = cipher.doFinal(authKey.getBytes());

        return Base64.getEncoder().encodeToString(encryptedBytes);
    }
}

