import java.security.PrivateKey;
import java.security.Signature;
import java.util.Arrays;

public class BAD_unsafeCheckSignatures {
    public boolean unsafeCheckSignatures(byte[] expected, byte[] data, PrivateKey key) throws Exception {
        Signature engine = Signature.getInstance("SHA256withRSA");
        engine.initSign(key);
        engine.update(data);
        byte[] signature = engine.sign();
        return Arrays.equals(expected, signature);
    }
}
