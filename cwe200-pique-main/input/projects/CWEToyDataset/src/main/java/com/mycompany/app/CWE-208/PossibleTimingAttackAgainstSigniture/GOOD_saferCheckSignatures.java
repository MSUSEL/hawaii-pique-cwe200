import java.security.MessageDigest;
import java.security.PrivateKey;
import java.security.Signature;

public class GOOD_saferCheckSignatures {
    public boolean saferCheckSignatures(byte[] expected, byte[] data, PrivateKey key) throws Exception {
        Signature engine = Signature.getInstance("SHA256withRSA");
        engine.initSign(key);
        engine.update(data);
        byte[] signature = engine.sign();
        return MessageDigest.isEqual(expected, signature);
    }
}
