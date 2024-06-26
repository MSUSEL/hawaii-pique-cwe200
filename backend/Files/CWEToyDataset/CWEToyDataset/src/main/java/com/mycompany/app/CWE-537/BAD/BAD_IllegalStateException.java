
public class BAD_IllegalStateException {

    public void logAttempt(String username, String password, boolean success) throws IllegalStateException{
        
        if (username == null || password == null) {
            throw new IllegalStateException("Login attempt for " + username + " with password " + password + " was " + (success ? "successful" : "unsuccessful"));
        }
    }
}
