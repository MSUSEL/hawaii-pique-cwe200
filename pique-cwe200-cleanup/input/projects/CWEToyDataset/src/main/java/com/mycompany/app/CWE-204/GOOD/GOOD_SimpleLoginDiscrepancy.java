public class GOOD_SimpleLoginDiscrepancy {
    public void login(String username, String password) {
        if (username.equals("admin")) {
            if (password.equals("password")) {
                System.out.println("Login Successful");
            } else {
                System.out.println("Login unsuccessful");
            }
        } else {
            System.out.println("Login unsuccessful");
        }
    }
}