public class BAD_VerboseAuthenticationFailure {

    public static void main(String[] args) {
        String username = "user";
        String password = "guessme";

        if (!authenticate(username, password)) {
            System.out.println("Authentication failed: " + password + " is incorrect or username does not exist.");
        }
    }

    private static boolean authenticate(String username, String password) {
        return false;
    }
}
