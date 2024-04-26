import org.apache.commons.logging.*;

public class GOOD_FinancialTransactionLogger {
    private static final Log logger = LogFactory.getLog(GOOD_FinancialTransactionLogger.class);

    public void logTransaction(String transactionId, double amount, String bankAccountNumber) {
        // Log transaction
        logger.info("Transaction logged."); 
    }
}
