public class GOOD_EmailVerificationCondition {
    public void verifyEmail(String email) {
        if (email != null) {
            if (email.equals("test@example.com")) {
                System.out.println("Verification Successful: Email found in system");
            } else {
                System.out.println("Verification Failed");
            }
        } else {
            System.out.println("Verification Failed");
        }
    }
}