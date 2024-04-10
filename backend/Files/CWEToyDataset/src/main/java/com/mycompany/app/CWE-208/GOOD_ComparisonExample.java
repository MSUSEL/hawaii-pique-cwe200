import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;

public class GOOD_ComparisonExample {
    
    public boolean secureTokenCompare(String token1, String token2) {
        byte[] tokenBytes1 = token1.getBytes(StandardCharsets.UTF_8);
        byte[] tokenBytes2 = token2.getBytes(StandardCharsets.UTF_8);
        
        return MessageDigest.isEqual(tokenBytes1, tokenBytes2);
    }
}
