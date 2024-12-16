import java.io.InputStream;
import java.net.Socket;
import java.util.Arrays;
import javax.crypto.Cipher;
import java.security.Key;

public class GOOD_AAS_noUserInputWhenCheckingCiphertext {
    // GOOD: compare ciphertexts using a constant-time method
    public boolean noUserInputWhenCheckingCiphertext(Socket socket, Key key) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] plaintext = readBytes(is, 100);
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, key);
            byte[] tag = cipher.doFinal(plaintext);
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