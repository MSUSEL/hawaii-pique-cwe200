import java.io.InputStream;
import java.net.Socket;
import java.util.Arrays;
import javax.crypto.Mac;


public class BAD_unsafeMacCheckWithLoop {
    public boolean unsafeMacCheckWithLoop(Socket socket) throws Exception {
        try (InputStream is = socket.getInputStream()) {
            byte[] data = readBytes(is, 256);
            byte[] tag = readBytes(is, 32);

            Mac mac = Mac.getInstance("Hmac256");
            byte[] computedTag = mac.doFinal(data);

            for (int i = 0; i < computedTag.length; i++) {
                byte a = computedTag[i];
                byte b = tag[i];
                if (a != b) {
                    return false;
                }
            }

            return true;
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
