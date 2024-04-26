import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class BAD_FinancialTransactionAPI {

    @PostMapping("/api/transaction")
    public String processTransaction(@RequestParam("accountNumber") String accountNumber, @RequestParam("amount") double amount) {
        try {
            // Process transaction that may include sensitive financial operations
            executeTransaction(accountNumber, amount);
            return "Transaction processed successfully";
        } catch (Exception e) {
            // Insecure: Exposing detailed transaction handling errors
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to process transaction for account " + accountNumber + " with amount $" + amount + ": " + e.getMessage(), e);
        }
    }

    private void executeTransaction(String accountNumber, double amount) throws Exception {
        // Simulate a failure that could arise during transaction processing
        if (amount > 10000) {
            throw new Exception("Transaction limit exceeded for single transaction.");
        }
        throw new Exception("Insufficient funds in account " + accountNumber);
    }
}
