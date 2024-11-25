import javax.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

public class GOOD_AttackAgainstHeader_safeComparison {
    public boolean safeComparison(HttpServletRequest request) {
        String token = request.getHeader("X-Auth-Token");
        String Key = "b1946ac92492d2347c6235b4d2611184";
        return MessageDigest.isEqual(Key.getBytes(StandardCharsets.UTF_8), token.getBytes(StandardCharsets.UTF_8));
    }
}
