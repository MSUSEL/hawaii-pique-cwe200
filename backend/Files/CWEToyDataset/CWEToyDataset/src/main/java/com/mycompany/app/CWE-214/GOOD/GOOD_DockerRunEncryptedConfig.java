import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_DockerRunEncryptedConfig {
    public static void main(String[] args) {
        try {
            String configSecret = "config123Secret!";
            String encryptedSecret = encryptSecret(configSecret);

            // We now pass the encrypted secret as an environment variable to the Docker process
            ProcessBuilder builder = new ProcessBuilder(
                "docker", "run", 
                "-e", "CONFIG_SECRET=" + encryptedSecret, 
                "myapp:latest"
            );
            builder.inheritIO();  // Inherit IO to see Docker output in Java process
            Process process = builder.start();
            process.waitFor();  // Wait for the process to complete
            System.out.println("Docker container started successfully with encrypted config.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Encrypt the configuration secret using AES encryption
    private static String encryptSecret(String secret) throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(256); // Use AES-256 for encryption
        SecretKey secretKey = keyGen.generateKey();

        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encryptedBytes = cipher.doFinal(secret.getBytes());

        // Encode bytes to Base64 to ensure safe transmission over environments like command lines
        return Base64.getEncoder().encodeToString(encryptedBytes);
    }
}
