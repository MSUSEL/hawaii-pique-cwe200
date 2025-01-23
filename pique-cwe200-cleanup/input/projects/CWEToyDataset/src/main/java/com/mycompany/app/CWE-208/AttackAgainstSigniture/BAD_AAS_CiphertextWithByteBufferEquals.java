import java.io.InputStream;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.security.Key;

import javax.crypto.Cipher;
import java.util.Arrays;

public class BAD_AAS_CiphertextWithByteBufferEquals {
    public boolean unsafeCheckCiphertextWithByteBufferEquals(Socket socket, Key key) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] plaintext = readBytes(is, 1024); // Assuming the data length to read
            Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
            cipher.init(Cipher.ENCRYPT_MODE, key);
            ByteBuffer tag = ByteBuffer.allocate(1024);
            cipher.doFinal(ByteBuffer.wrap(plaintext), tag);
            byte[] expected = readBytes(is, 32);
            return ByteBuffer.wrap(expected).equals(tag);
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
