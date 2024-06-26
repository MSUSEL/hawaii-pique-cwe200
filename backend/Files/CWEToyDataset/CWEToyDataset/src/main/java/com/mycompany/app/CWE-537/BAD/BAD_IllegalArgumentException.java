public class BAD_IllegalArgumentException {

    public void registerUser(String username, String email, String password) throws IllegalArgumentException{
        
        if (!email.contains("@")) {
            throw new IllegalArgumentException("An error occurred while registering user with email: " + email + ". Please provide a valid email address.");
        }
        else{
            System.out.println("User registered successfully.");
        }
    }
}
