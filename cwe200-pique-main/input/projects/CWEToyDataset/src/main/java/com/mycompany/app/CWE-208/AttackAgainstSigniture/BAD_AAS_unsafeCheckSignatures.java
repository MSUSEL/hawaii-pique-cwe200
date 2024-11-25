import java.io.InputStream;
import java.net.Socket;
import java.security.PrivateKey;
import java.security.Signature;
import java.util.Arrays;

public class BAD_AAS_unsafeCheckSignatures {
    public boolean unsafeCheckSignatures(Socket socket, PrivateKey key) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            Signature engine = Signature.getInstance("SHA256withRSA");
            engine.initSign(key);
            byte[] data = readBytes(is, 1024); // Assuming data to read is 1024 bytes
            engine.update(data);
            byte[] signature = engine.sign();
            byte[] expected = readBytes(is, 256);
            return Arrays.equals(expected, signature);
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
