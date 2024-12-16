public class BAD_FinancialAuditError {
    public void processAuditReport(String financialYear, double declaredRevenue) {
        try {
            if (declaredRevenue < 0) {
                throw new Exception("Revenue cannot be negative: " + declaredRevenue);
            }
            // Audit processing logic
        } catch (Exception e) {
            System.err.println("Audit processing error for financial year " + financialYear + ": " + e.getMessage());
            throw new RuntimeException("Failed to process audit for financial year " + financialYear, e);
        }
    }

    public static void main(String[] args) {
        new BAD_FinancialAuditError().processAuditReport("2023", -500000.00);
    }
}
