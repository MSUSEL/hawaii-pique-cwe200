public class GOOD_EncryptionKeyError {
    public void useEncryptionKey(String key, String data) {
        try {
            if (key.length() < 32) {
                // Use a generic error message that does not reveal details about the key
                throw new IllegalArgumentException("Invalid encryption key.");
            }
            // Encryption logic
        } catch (IllegalArgumentException e) {
            // Log the error internally without exposing it to the user or external logs
            System.err.println("Failed to initialize encryption.");
            // Throw a generic SecurityException without specific details
            throw new SecurityException("Encryption initialization failed.");
        }
    }

    public static void main(String[] args) {
        new GOOD_EncryptionKeyError().useEncryptionKey("12345", "Sensitive data to encrypt");
    }
}
