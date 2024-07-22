import java.security.Key;
import javax.crypto.Cipher;
import java.util.Arrays;

public class BAD_unsafeCheckCustomMac {
    public boolean unsafeCheckCustomMac(byte[] expected, byte[] plaintext, Key key) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, key);
        byte[] tag = cipher.doFinal(plaintext);
        return Arrays.equals(expected, tag);
    }
}
