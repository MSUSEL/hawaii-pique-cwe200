// Define a custom checked exception
class InvalidAccessLevelException extends Exception {
    public InvalidAccessLevelException(String message) {
        super(message);
    }
}

public class BAD_AccessControlError {
    public void validateUserAccess(String userName, int accessLevel) throws InvalidAccessLevelException {
        if (accessLevel < 1 || accessLevel > 5) {
            throw new InvalidAccessLevelException(
                "Invalid access level " + accessLevel + " for user: " + userName + 
                ". Access level must be between 1 and 5."
            );
        }
        // Access validation logic
        System.out.println("Access validated for user: " + userName + " with level: " + accessLevel);
    }

    public static void main(String[] args) {
        BAD_AccessControlError accessControl = new BAD_AccessControlError();
        try {
            accessControl.validateUserAccess("adminUser", 0);
        } catch (InvalidAccessLevelException e) {
            System.err.println("Access Control Error: " + e.getMessage());
        }
    }
}
