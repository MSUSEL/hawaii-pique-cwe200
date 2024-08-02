public class BAD_IllegalArgumentException {

    public void registerUser(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException();
            } else {
                System.out.println("User registered successfully.");
            }
        } catch (IllegalArgumentException e) {
            System.err.println("An error occurred while registering user with email: " + email + ". Please provide a valid email address.");
        }
    }

}
