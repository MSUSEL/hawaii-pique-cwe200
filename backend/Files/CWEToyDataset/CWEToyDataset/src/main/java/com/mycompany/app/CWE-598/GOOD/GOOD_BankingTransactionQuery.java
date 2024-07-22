import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_BankingTransactionQuery {
    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // AES 128-bit key
            secretKey = keyGen.generateKey();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static String encryptAccountNumber(String accountNumber) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(accountNumber.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void queryTransactionDetails(String accountNumber) {
        try {
            accountNumber = encryptAccountNumber(accountNumber);
            String url = "https://example.com/transactions"; // Use HTTPS
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpPost httpPost = new HttpPost(url);

            // Setting the Request headers
            httpPost.setHeader("Content-Type", "application/json");
            // Construct JSON payload
            String jsonPayload = "{\"accountNumber\":\"" + accountNumber + "\"}";
            StringEntity entity = new StringEntity(jsonPayload);
            httpPost.setEntity(entity);

            HttpResponse response = httpClient.execute(httpPost);
            httpClient.close();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        queryTransactionDetails("1234567890"); 
    }
}
