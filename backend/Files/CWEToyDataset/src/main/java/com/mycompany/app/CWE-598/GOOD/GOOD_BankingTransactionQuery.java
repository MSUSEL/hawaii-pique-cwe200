import org.apache.http.client.methods.HttpGet;
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
            e.printStackTrace();
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
            String url = "http://example.com/transactions?accountNumber=" + accountNumber; // Encrypted data in query
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);

            HttpResponse response = httpClient.execute(httpGet);
            System.out.println("Transaction details requested for encrypted account number: " + accountNumber);
            System.out.println("Response Code : " + response.getStatusLine().getStatusCode());

            httpClient.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        queryTransactionDetails("1234567890"); // Now with encrypted account number
    }
}
