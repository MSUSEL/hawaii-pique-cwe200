public class BAD_TwoFactorAuthentication {
    public void authenticate(String username, String code) {
        if (username.equals("admin")) {
            if (code.equals("123456")) {
                System.out.println("Authentication Successful");
            } else {
                System.out.println("Authentication Failed: Invalid code for admin");
            }
        } else {
            System.out.println("Authentication Failed: Unknown username");
        }
    }
}
