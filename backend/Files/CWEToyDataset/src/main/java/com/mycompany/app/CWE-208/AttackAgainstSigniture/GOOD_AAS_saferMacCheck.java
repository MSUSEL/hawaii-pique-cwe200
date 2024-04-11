import java.io.InputStream;
import java.net.Socket;
import javax.crypto.Mac;
import java.security.MessageDigest;
import java.util.Arrays;

public class GOOD_AAS_saferMacCheck {
    public boolean saferMacCheck(Socket socket) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] data = new byte[1024];
            int readBytes = is.read(data);
      
            if (readBytes < data.length) {
                data = Arrays.copyOf(data, readBytes);
            }
            Mac mac = Mac.getInstance("HmacSHA256");
            byte[] actualMac = mac.doFinal(data);

            byte[] expectedMac = readBytes(is, 32);
            return MessageDigest.isEqual(expectedMac, actualMac);
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
