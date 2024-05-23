import java.io.InputStream;
import java.net.Socket;
import java.util.Arrays;

import javax.crypto.Mac;

public class GOOD_AAS_compareMacWithConstant {
    // GOOD: compare MAC with constant using a constant-time method
    public boolean compareMacWithConstant(Socket socket) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            Mac mac = Mac.getInstance("HmacSHA256");
            byte[] data = readBytes(is, 1024);
            byte[] actualMac = mac.doFinal(data);
            return "constant".equals(new String(actualMac));
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

