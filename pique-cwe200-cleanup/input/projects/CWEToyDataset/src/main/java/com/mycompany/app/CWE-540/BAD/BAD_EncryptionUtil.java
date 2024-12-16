import java.security.PrivateKey;
import java.security.KeyFactory;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;

public class BAD_EncryptionUtil {
    private static final String PRIVATE_KEY = "-----BEGIN PRIVATE KEY-----MIICeAIBADANBgkqhkiG9w0BAQEFAASC... -----END PRIVATE KEY-----"; 

    public PrivateKey getPrivateKey() throws Exception {
        byte[] keyBytes = Base64.getDecoder().decode(PRIVATE_KEY.getBytes());
        PKCS8EncodedKeySpec spec = new PKCS8EncodedKeySpec(keyBytes);
        KeyFactory kf = KeyFactory.getInstance("RSA");
        return kf.generatePrivate(spec);
    }
}
