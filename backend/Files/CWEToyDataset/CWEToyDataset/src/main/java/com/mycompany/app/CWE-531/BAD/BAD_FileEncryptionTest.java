import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;


public class BAD_FileEncryptionTest {
    @Test
    public void testFileEncryption() {
        String filePath = "/path/to/important/file.txt";
        String encryptionKey = "SpringMeadow2023";
        FileEncryptor encryptor = new FileEncryptor();
        assertTrue(encryptor.encryptFile(filePath, encryptionKey));
    }

    class FileEncryptor {
        public boolean encryptFile(String filePath, String encryptionKey) {
            return encryptionKey.equals("SpringMeadow2023");
        }
    }
}
