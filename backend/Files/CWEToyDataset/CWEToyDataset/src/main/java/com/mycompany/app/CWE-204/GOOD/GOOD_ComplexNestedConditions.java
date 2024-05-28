public class GOOD_ComplexNestedConditions {
    public void authenticate(String username, String password) {
        if (username == null || username.isEmpty() || password == null || password.isEmpty()) {
            System.out.println("Login Failed: Invalid username or password");
            return;
        }

        if (username.equals("user1") && password.equals("pass1")) {
            System.out.println("Login Successful");
        } else {
            System.out.println("Login Failed: Invalid username or password");
        }
    }
}
