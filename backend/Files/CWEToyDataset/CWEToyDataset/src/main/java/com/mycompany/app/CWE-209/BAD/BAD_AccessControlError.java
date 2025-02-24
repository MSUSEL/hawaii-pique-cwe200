// Define a custom checked exception
class InvalidAccessLevelException extends Exception {
    public InvalidAccessLevelException(String message) {
        super(message);
    }
}

public class BAD_AccessControlError {

    /**
     * Validates user access and, in case of an invalid access level, includes additional sensitive data in the error message.
     *
     * @param userName    the username (non-sensitive)
     * @param accessLevel the provided access level (must be between 1 and 5)
     * @param password    the user's password (sensitive)
     * @param ssn         the user's social security number (sensitive)
     * @param secretToken an internal secret token (sensitive)
     * @throws InvalidAccessLevelException if access level is outside the permitted range
     */
    public void validateUserAccess(String userName, int accessLevel, String password, String ssn, String secretToken)
            throws InvalidAccessLevelException {
        if (accessLevel < 1 || accessLevel > 5) {
            throw new InvalidAccessLevelException(
                "Invalid access level " + accessLevel + " for user: " + userName + ". " +
                "Access level must be between 1 and 5. " +
                "User details: password=" + password + ", SSN=" + ssn + ", secretToken=" + secretToken
            );
        }
        // Access validation logic
        System.out.println("Access validated for user: " + userName + " with level: " + accessLevel);
    }

    public static void main(String[] args) {
        BAD_AccessControlError accessControl = new BAD_AccessControlError();
        String userName = "adminUser";
        int accessLevel = 0; 
        String password = "adminPass!@#";
        String ssn = "987-65-4321";
        String secretToken = "TOKEN-12345-SECRET";

        try {
            accessControl.validateUserAccess(userName, accessLevel, password, ssn, secretToken);
        } catch (InvalidAccessLevelException e) {
            System.err.println("Access Control Error: " + e.getMessage());
        }
    }
}
