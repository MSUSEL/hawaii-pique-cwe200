import java.io.InputStream;
import java.net.Socket;
import javax.crypto.Mac;
import java.util.Arrays;

public class BAD_AAS_unsafeMacCheckWithDoFinalWithOutputArray {
    public boolean unsafeMacCheckWithDoFinalWithOutputArray(Socket socket) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] data = readBytes(is, 100);
            Mac mac = Mac.getInstance("HmacSHA256");
            byte[] actualMac = new byte[256];
            mac.update(data);
            mac.doFinal(actualMac, 0);
            byte[] expectedMac = readBytes(is, 256);
            return Arrays.equals(expectedMac, actualMac);
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
