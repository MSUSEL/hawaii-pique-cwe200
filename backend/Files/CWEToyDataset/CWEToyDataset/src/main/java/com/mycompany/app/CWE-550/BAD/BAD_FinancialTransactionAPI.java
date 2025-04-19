import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@RestController
public class BAD_FinancialTransactionAPI {

    @PostMapping("/api/transaction")
    public String processTransaction(@RequestParam("accountNumber") String accountNumber,
                                     @RequestParam("amount") double amount) {
        try {
            executeTransaction(accountNumber, amount);
            return "Transaction processed successfully";
        } catch (ResponseStatusException e) {
            return "Failed to process transaction for account " + accountNumber +
                   " with amount $" + amount + ": " + e.getMessage();
        }
    }

    private void executeTransaction(String accountNumber, double amount) throws ResponseStatusException {
        if (amount > 10000) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "Transaction limit exceeded for account " + accountNumber + " with amount $" + amount);
        }
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
            "Insufficient funds in account " + accountNumber + " for amount $" + amount);
    }
}