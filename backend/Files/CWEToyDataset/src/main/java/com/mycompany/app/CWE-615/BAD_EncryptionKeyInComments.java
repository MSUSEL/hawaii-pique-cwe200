public class BAD_EncryptionKeyInComments {
    /*
     * Encryption service for sensitive user data.
     * NOTE: Encryption key for development - AESKey123456789 (Replace before production)
     */
    public static void encryptUserData(String userData) {
        // Placeholder for encryption logic using the mentioned key
    }
    
    public static void main(String[] args) {
        encryptUserData("Sensitive user data");
    }
}
