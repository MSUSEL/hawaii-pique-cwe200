
import java.security.Key;
import java.security.MessageDigest;
import javax.crypto.Cipher;

public class GOOD_saferCheckCustomMac {
    public boolean saferCheckCustomMac(byte[] expected, byte[] plaintext, Key key) throws Exception {
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, key);
        byte[] tag = cipher.doFinal(plaintext);
        return MessageDigest.isEqual(expected, tag);
    }
}
