public class BAD_ComplexNestedConditions {
    public void authenticate(String username, String password) {
        if (username != null && !username.isEmpty()) {
            if (password != null && !password.isEmpty()) {
                if (username.equals("user1")) {
                    if (password.equals("pass1")) {
                        System.out.println("Login Successful");
                    } else {
                        System.out.println("Incorrect password");
                    }
                } else {
                    System.out.println("User not found");
                }
            } else {
                System.out.println("Password cannot be empty");
            }
        } else {
            System.out.println("Username cannot be empty");
        }
    }
}
