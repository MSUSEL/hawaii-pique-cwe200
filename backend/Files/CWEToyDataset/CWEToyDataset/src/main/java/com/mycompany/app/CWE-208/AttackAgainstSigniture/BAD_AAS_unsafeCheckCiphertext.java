import java.io.InputStream;
import java.net.Socket;
import java.security.Key;
import java.security.MessageDigest;
import java.util.Objects;
import javax.crypto.Cipher;
import java.util.Arrays;
public class BAD_AAS_unsafeCheckCiphertext {
    public boolean unsafeCheckCiphertext(Socket socket, Key key) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] plaintext = readBytes(is, 100);
            byte[] hash = MessageDigest.getInstance("SHA-256").digest(plaintext);
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, key);
            byte[] tag = cipher.doFinal(hash);
            byte[] expected = readBytes(is, tag.length); // Assume expected length is same as tag length
            return Objects.deepEquals(expected, tag);
        }
    }

    private byte[] readBytes(InputStream is, int numBytes) throws Exception {
        byte[] bytes = new byte[numBytes];
        int offset = 0;
        int read;
        while (offset < numBytes && (read = is.read(bytes, offset, numBytes - offset)) != -1) {
            offset += read;
        }
        if (offset < numBytes) {
            return Arrays.copyOf(bytes, offset);
        }
        return bytes;
    }
}
