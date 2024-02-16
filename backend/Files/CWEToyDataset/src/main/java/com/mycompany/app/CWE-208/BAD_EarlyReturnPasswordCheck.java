public class BAD_EarlyReturnPasswordCheck {
    // Vulnerability: The method returns as soon as a mismatch is found, 
    // leading to a timing discrepancy based on how much of the input password matches the actual password.

    public static boolean checkPassword(String inputPassword, String actualPassword) {
        if (inputPassword.length() != actualPassword.length()) {
            return false; // Early return on length mismatch
        }
        for (int i = 0; i < inputPassword.length(); i++) {
            if (inputPassword.charAt(i) != actualPassword.charAt(i)) {
                return false; // Early return on the first mismatch
            }
        }
        return true;
    }

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        checkPassword("inputPass", "secretPass");
        long endTime = System.currentTimeMillis();
        System.out.println("Password check took: " + (endTime - startTime) + "ms");
    }
}
