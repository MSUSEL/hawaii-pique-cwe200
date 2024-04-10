import javax.servlet.http.HttpServletRequest;
import java.security.MessageDigest;
import java.nio.charset.StandardCharsets;

public class GOOD_TimingAttackAgainstHeader {
    
    private boolean safeComparison(HttpServletRequest request) {
          String token = request.getHeader("X-Auth-Token");
          String Key = "secret"; 
          return MessageDigest.isEqual(Key.getBytes(StandardCharsets.UTF_8), token.getBytes(StandardCharsets.UTF_8));
    }
    
}