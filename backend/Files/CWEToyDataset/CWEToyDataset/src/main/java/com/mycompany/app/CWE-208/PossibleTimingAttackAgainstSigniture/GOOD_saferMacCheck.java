import javax.crypto.Mac;
import java.security.MessageDigest;

public class GOOD_saferMacCheck {
    public boolean saferMacCheck(byte[] expectedMac, byte[] data) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        byte[] actualMac = mac.doFinal(data);
        return MessageDigest.isEqual(expectedMac, actualMac);
    }
}
