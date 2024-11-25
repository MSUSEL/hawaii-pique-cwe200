public class BAD_EncryptionKeyError {
    public void useEncryptionKey(String key, String data) {
        try {
            if (key.length() < 32) {
                throw new IllegalArgumentException("Encryption key too short: " + key);
            }
            // Encryption logic
        } catch (IllegalArgumentException e) {
            System.err.println("Encryption error: " + e.getMessage());
            throw new SecurityException("Encryption failed due to inadequate key length for data: " + data);
        }
    }

    public static void main(String[] args) {
        new BAD_EncryptionKeyError().useEncryptionKey("12345", "Sensitive data to encrypt");
    }
}
