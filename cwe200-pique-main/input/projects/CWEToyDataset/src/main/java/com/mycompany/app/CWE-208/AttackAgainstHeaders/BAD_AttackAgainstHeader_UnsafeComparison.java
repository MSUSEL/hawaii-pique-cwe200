import javax.servlet.http.HttpServletRequest;

public class BAD_AttackAgainstHeader_UnsafeComparison {
    public boolean UnsafeComparison(HttpServletRequest request) {
        String Key = "5d41402abc4b2a76b9719d911017c592";
        return Key.equals(request.getHeader("X-Auth-Token"));
    }
}
