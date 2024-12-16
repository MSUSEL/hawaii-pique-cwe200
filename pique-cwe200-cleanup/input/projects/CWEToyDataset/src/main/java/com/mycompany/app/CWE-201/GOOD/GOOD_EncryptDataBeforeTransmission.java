import javax.crypto.Cipher;
import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class GOOD_EncryptDataBeforeTransmission {
    private static final String KEY = "Bar12345Bar12345";
    
    public static String encryptData(String data) throws Exception {
        SecretKeySpec dataKey = new SecretKeySpec(KEY.getBytes(), "AES");
        Cipher cipher = Cipher.getInstance("AES/ECB/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, dataKey);
        
        byte[] encryptedData = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encryptedData);
    }

    public static void main(String[] args) {
        try {
            String APIKey = "The API token is 123";
            String encryptedData = encryptData(APIKey);
            System.out.println("Encrypted Data: " + encryptedData);
        } catch (Exception e) {
            System.err.println("An error has occurred.");
        }
    }
}
