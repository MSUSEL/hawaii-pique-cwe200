import org.apache.commons.logging.*;

public class BAD_FinancialTransactionLogger {
    private static final Log logger = LogFactory.getLog(BAD_FinancialTransactionLogger.class);

    public void logTransaction(String transactionId, double amount, String bankAccountNumber) {
        // Log transaction details
        logger.info("Transaction logged with ID: " + transactionId + ", amount: " + amount + ", account number: " + bankAccountNumber); // Logging sensitive bank account number
    }
}
