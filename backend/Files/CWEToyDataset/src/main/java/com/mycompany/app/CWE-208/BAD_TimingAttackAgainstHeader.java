import javax.servlet.http.HttpServletRequest;
import java.lang.String;

public class BAD_TimingAttackAgainstHeader {
    private boolean UnsafeComparison(HttpServletRequest request) {
        String Key = "secret";
        return Key.equals(request.getHeader("X-Auth-Token"));        
    }    
}

