import java.io.*;

public class GOOD_SecretFileDecryptor {
    public static void decryptFile(String decryptionToolPath, String encryptedFilePath) {
        try {
            Process process = Runtime.getRuntime().exec(decryptionToolPath + " " + encryptedFilePath);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error = errorReader.readLine();

            if (process.waitFor() != 0) {
                // Exposing decryption failure details that could hint at file paths or encryption methods
                System.err.println("Decryption failed.");
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Decryption process error.");
        }
    }
}