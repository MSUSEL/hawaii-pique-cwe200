public class BAD_NetworkAuthenticationConditional {

    public void authenticate(String username, String password) {
        if (username != null && password != null) {
            if (username.equals("admin") && password.equals("adminPass")) {
                System.out.println("Authentication Successful: Admin access granted.");
            } else if (username.equals("admin")) {
                System.out.println("Authentication Failed: Incorrect admin password.");
            } else if (password.equals("userPass")) {
                System.out.println("Authentication Failed: Incorrect username.");
            } else {
                System.out.println("Authentication Failed: Invalid credentials.");
            }
        } else {
            System.out.println("Authentication Failed: Missing username or password.");
        }
    }
}