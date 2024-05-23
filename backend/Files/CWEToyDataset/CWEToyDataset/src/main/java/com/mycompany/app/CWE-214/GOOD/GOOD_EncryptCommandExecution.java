import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;

public class GOOD_EncryptCommandExecution {
    public static void main(String[] args) {
        try {
            String sensitiveArgument = "sensitiveValue";
            String encryptedArgument = encryptArgument(sensitiveArgument);

            ProcessBuilder builder = new ProcessBuilder("secureScript", encryptedArgument);
            Process process = builder.start();
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String encryptArgument(String argument) throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(128);
        SecretKey secretKey = keyGen.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(argument.getBytes());

        return Base64.getEncoder().encodeToString(encrypted);
    }
}
