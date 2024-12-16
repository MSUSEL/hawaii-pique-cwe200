public class GOOD_VerboseAuthenticationFailure {

    public static void main(String[] args) {
        String username = "user";
        String password = "guessme";

        if (!authenticate(username, password)) {
            System.out.println("Authentication failed: Please check your username and password.");
        }
    }

    private static boolean authenticate(String username, String password) {
        return false;
    }
}