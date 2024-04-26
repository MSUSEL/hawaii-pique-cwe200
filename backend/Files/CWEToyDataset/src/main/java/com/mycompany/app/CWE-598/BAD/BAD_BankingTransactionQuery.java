import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;

public class BAD_BankingTransactionQuery {
    public static void queryTransactionDetails(String accountNumber) {
        try {
            String url = "http://example.com/transactions?accountNumber=" + accountNumber; // Sensitive data in query
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);

            HttpResponse response = httpClient.execute(httpGet);
            System.out.println("Transaction details requested for account number: " + accountNumber);
            System.out.println("Response Code : " + response.getStatusLine().getStatusCode());

            httpClient.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        queryTransactionDetails("1234567890"); // Extremely risky!
    }
}
