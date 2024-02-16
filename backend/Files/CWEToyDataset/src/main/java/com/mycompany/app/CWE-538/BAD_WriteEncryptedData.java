import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class BAD_WriteEncryptedData {
    public static void main(String[] args) {
        String encryptedData = "5f4dcc3b5aa765d61d8327deb882cf99"; // Example of encrypted sensitive data

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("/shared/encryptedData.txt"))) {
            // Insecure: Writing encrypted sensitive data to a shared directory that might be externally accessible
            writer.write("Encrypted data: " + encryptedData);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
