import javax.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

public class GOOD_AttackAgainstHeader_safeComparison {
    public boolean safeComparison(HttpServletRequest request) {
        String token = request.getHeader("X-Auth-Token");
        String Key = "secret";
        return MessageDigest.isEqual(Key.getBytes(StandardCharsets.UTF_8), token.getBytes(StandardCharsets.UTF_8));
    }
}
