public class BAD_SimpleLoginDiscrepancy {
    public void login(String username, String password) {
        if (username.equals("admin")) {
            if (password.equals("admin")) {
                System.out.println("Login Successful");
            } else {
                System.out.println("Invalid password");
            }
        } else {
            System.out.println("Invalid username");
        }
    }
}
