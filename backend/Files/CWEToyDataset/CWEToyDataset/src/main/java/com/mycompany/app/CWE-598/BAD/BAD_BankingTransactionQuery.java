import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;

public class BAD_BankingTransactionQuery {
    public static void queryTransactionDetails(String accountNumber) {
        try {
            String url = "http://services.bankcorp.com/transactions?accountNumber" + accountNumber;
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);
            HttpResponse response = httpClient.execute(httpGet);

            httpClient.close();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        queryTransactionDetails("1234567890"); 
    }
}
