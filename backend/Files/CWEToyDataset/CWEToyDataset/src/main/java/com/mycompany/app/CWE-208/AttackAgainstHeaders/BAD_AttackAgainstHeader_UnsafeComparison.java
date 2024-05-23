import javax.servlet.http.HttpServletRequest;

public class BAD_AttackAgainstHeader_UnsafeComparison {
    public boolean UnsafeComparison(HttpServletRequest request) {
        String Key = "secret";
        return Key.equals(request.getHeader("X-Auth-Token"));
    }
}
