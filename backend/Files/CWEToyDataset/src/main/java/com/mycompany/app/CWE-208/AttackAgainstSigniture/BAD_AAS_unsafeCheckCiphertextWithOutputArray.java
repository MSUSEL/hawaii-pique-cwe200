import java.io.InputStream;
import java.net.Socket;
import java.security.Key;
import java.security.MessageDigest;
import javax.crypto.Cipher;
import java.util.Arrays;

public class BAD_AAS_unsafeCheckCiphertextWithOutputArray {
    public boolean unsafeCheckCiphertextWithOutputArray(Socket socket, Key key) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] plaintext = readBytes(is, 1024); // Assuming the data length to read
            MessageDigest md = MessageDigest.getInstance("SHA-512");
            md.update(plaintext);
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, key);
            byte[] tag = new byte[1024];
            cipher.doFinal(tag, 0);
            byte[] expected = readBytes(is, 32);
            return Arrays.equals(expected, tag);
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
