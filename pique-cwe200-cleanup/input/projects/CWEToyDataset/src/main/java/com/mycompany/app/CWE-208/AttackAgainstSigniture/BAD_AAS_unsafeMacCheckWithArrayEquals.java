import java.io.InputStream;
import java.net.Socket;
import javax.crypto.Mac;
import java.util.Arrays;

public class BAD_AAS_unsafeMacCheckWithArrayEquals {
    public boolean unsafeMacCheckWithArrayEquals(Socket socket) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            Mac mac = Mac.getInstance("HmacSHA256");
            byte[] data = new byte[1024];
            is.read(data);
            byte[] actualMac = mac.doFinal(data);
            byte[] expectedMac = new byte[32];
            int expectedReadBytes = is.read(expectedMac);
            if (expectedReadBytes < expectedMac.length) {
                expectedMac = Arrays.copyOf(expectedMac, expectedReadBytes);
            }
            return Arrays.equals(expectedMac, actualMac);
        }
    }
}
