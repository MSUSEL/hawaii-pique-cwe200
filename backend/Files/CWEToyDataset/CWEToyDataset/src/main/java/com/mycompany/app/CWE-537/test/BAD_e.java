public class BAD_e {
    
    public void registerUser2(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException("Invalid credentials provided." + email + password + username);
            } else {
                System.out.println("User registered successfully.");
            }
        } catch (IllegalArgumentException e) {
            System.err.println(e);
        }
    }
}
