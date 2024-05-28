public class BAD_EmailVerificationCondition {
    public void verifyEmail(String email) {
        if (email.equals("test@example.com")) {
            System.out.println("Verification Successful: Email found in system");
        } else {
            System.out.println("Verification Failed: Email not registered");
        }
    }
}
