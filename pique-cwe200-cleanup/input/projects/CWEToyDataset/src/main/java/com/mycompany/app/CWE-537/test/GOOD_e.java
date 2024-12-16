public class GOOD_e {
    public void registerUser3(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException("Invalid credentials provided.");
            } else {
                System.out.println("User registered successfully.");
            }
        } catch (IllegalArgumentException e) {
            System.err.println(e);
        }
    }
}
