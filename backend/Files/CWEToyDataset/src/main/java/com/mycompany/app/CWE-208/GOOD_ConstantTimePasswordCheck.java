public class GOOD_ConstantTimePasswordCheck {
    // Mitigation: Performing password comparison in constant time to prevent timing attacks,
    // ensuring the operation takes the same amount of time regardless of input.

    public static boolean constantTimeCompare(String inputPassword, String actualPassword) {
        boolean result = true;
        int maxLength = Math.max(inputPassword.length(), actualPassword.length());
        for (int i = 0; i < maxLength; i++) {
            char inputChar = i < inputPassword.length() ? inputPassword.charAt(i) : 0;
            char actualChar = i < actualPassword.length() ? actualPassword.charAt(i) : 0;
            result &= (inputChar == actualChar);
        }
        return result && inputPassword.length() == actualPassword.length();
    }

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        constantTimeCompare("inputPass", "secretPass");
        long endTime = System.nanoTime();
        System.out.println("Password comparison took: " + (endTime - startTime) + "ns");
    }
}
