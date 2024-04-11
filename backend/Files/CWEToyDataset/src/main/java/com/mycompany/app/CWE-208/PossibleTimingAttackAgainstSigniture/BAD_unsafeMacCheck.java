
import javax.crypto.Mac;
import java.util.Arrays;
//PossibleTimingAttackAgainstSigniture
public class BAD_unsafeMacCheck {
    // BAD: compare MACs using a not-constant time method
    public boolean unsafeMacCheck(byte[] expectedMac, byte[] data) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        byte[] actualMac = mac.doFinal(data);
        return Arrays.equals(expectedMac, actualMac);
    }
}
