// Define a custom checked exception
class EncryptionKeyException extends Exception {
    public EncryptionKeyException(String message) {
        super(message);
    }
}

public class BAD_EncryptionKeyError {
    public void useEncryptionKey(String encryptionKey, String data) throws EncryptionKeyException {
        if (encryptionKey.length() < 32) {
            throw new EncryptionKeyException("Encryption key " + encryptionKey + " is too short. Key length must be at least 32 characters.");
        }
        // Encryption logic
        System.out.println("Data encrypted successfully.");
    }

    public static void main(String[] args) {
        BAD_EncryptionKeyError encryptionKeyError = new BAD_EncryptionKeyError();
        try {
            encryptionKeyError.useEncryptionKey("12345", "Sensitive data to encrypt");
        } catch (EncryptionKeyException e) {
            System.err.println("Encryption Error: " + e.getMessage());
            // Log sensitive information securely without exposing it
            System.err.println("Operation failed due to inadequate encryption key.");
        }
    }
}
