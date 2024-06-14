public class GOOD_IllegalArgumentException {

    public void registerUser(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException("Email format is incorrect.");
            }
            // Logic to register user
        } catch (IllegalArgumentException e) {
            System.err.println("Registration attempt failed.");
            throw new RuntimeException("Invalid email format.");
        }
    }
}
