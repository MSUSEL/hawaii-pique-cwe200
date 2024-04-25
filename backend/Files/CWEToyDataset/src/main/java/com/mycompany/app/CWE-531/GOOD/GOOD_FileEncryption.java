import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FileEncryptor {
    public boolean encryptFile(String filePath, String encryptionKey) {
        // Assume this method performs actual encryption and returns true if successful
        return encryptionKey.equals(System.getenv("ENCRYPTION_KEY"));
    }
}

public class GOOD_FileEncryption {
    @Test
    public void testFileEncryption() {
        String filePath = System.getenv("FILE_PATH");
        // Fetch encryption key from environment variables
        String encryptionKey = System.getenv("ENCRYPTION_KEY");
        if (encryptionKey == null || encryptionKey.isEmpty()) {
            throw new IllegalStateException("Encryption key must be set in the environment variables.");
        }

        FileEncryptor encryptor = new FileEncryptor();
        assertTrue(encryptor.encryptFile(filePath, encryptionKey), "File encryption should be successful.");
    }
}
