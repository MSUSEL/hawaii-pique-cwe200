public class GOOD_ComplexNestedConditions {
    public void authenticate(String username, String password) {
        if (username != null && !username.isEmpty()) {
            if (password != null && !password.isEmpty()) {
                if (username.equals("user1")) {
                    if (password.equals("pass1")) {
                        System.out.println("Login Successful");
                    } else {
                        System.out.println("Error");
                    }
                } else {
                    System.out.println("Error");
                }
            } else {
                System.out.println("Error");
            }
        } else {
            System.out.println("Error");
        }
    }
}
