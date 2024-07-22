public class GOOD_EncryptionKeyError {
    public void useEncryptionKey(String key, String data) {
        try {
            if (key.length() < 32) {
                throw new IllegalArgumentException("Invalid encryption key.");
            }
            // Encryption logic
        } catch (IllegalArgumentException e) {
            System.err.println("Failed to initialize encryption.");
            throw new SecurityException("Encryption initialization failed.");
        }
    }

    public static void main(String[] args) {
        new GOOD_EncryptionKeyError().useEncryptionKey("12345", "Sensitive data to encrypt");
    }
}
