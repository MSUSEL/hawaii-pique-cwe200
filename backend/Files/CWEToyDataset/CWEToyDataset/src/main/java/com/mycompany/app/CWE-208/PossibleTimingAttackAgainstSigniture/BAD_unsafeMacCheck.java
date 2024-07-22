
import javax.crypto.Mac;
import java.util.Arrays;

public class BAD_unsafeMacCheck {
    public boolean unsafeMacCheck(byte[] expectedMac, byte[] data) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        byte[] actualMac = mac.doFinal(data);
        return Arrays.equals(expectedMac, actualMac);
    }
}
