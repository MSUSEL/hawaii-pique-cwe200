import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_SecureProcessInvocationWithEncryption {
    public static void main(String[] args) {
        try {
            String sensitiveData = "VerySensitiveData";
            String encryptedData = encryptData(sensitiveData);

            ProcessBuilder builder = new ProcessBuilder("secureProcess", encryptedData);
            builder.start().waitFor();
            System.out.println("Secure process invoked successfully with encrypted data.");
        } catch (Exception e) {
            System.err.println("Error during secure process invocation.");
            e.printStackTrace();
        }
    }

    private static String encryptData(String data) throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(256);
        SecretKey secretKey = keyGen.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());

        return Base64.getEncoder().encodeToString(encrypted);
    }
}
