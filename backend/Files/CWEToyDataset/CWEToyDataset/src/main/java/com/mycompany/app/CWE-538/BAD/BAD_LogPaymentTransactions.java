import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class BAD_LogPaymentTransactions {
    // Simulates the insecure practice of logging sensitive transaction information directly to a file.
    private static final String TRANSACTION_LOG_FILE = "/var/log/app/paymentTransactions.log"; // A log file that might be improperly secured

    public static void logPaymentTransaction(String fullName, String creditCardNumber, double amount) {
        // Constructing the log message with sensitive transaction details
        String logMessage = String.format("Processing payment transaction: Full Name=%s, Credit Card Number=%s, Amount=%.2f",
                                          fullName, creditCardNumber, amount);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(TRANSACTION_LOG_FILE, true))) {
            // Writing the sensitive log message to the file
            writer.write(logMessage + System.lineSeparator());
        } catch (IOException e) {
            System.err.println("Error writing to log file: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Example usage of the insecure logging method
        logPaymentTransaction("Alice Johnson", "1234567890123456", 99.99);
    }
}
