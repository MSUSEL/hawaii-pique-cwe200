import javax.crypto.spec.SecretKeySpec;
import java.security.Key;
import java.util.Base64;

public class GOOD_CryptoKeyStorage {
    public Key getEncryptionKey() {
        String encodedKey = System.getenv("ENCRYPTION_KEY");
        byte[] decodedKey = Base64.getDecoder().decode(encodedKey);
        return new SecretKeySpec(decodedKey, "AES");
    }
}
