public class BAD_FinancialAuditError {
    public void processAuditReport(String financialYear, double declaredRevenue) {
        try {
            if (declaredRevenue < 0) {
                String dbConnectionString = "jdbc:mysql://prod-db:3306/finance?user=financeUser&password=FinPass!@#";
                String internalAuditKey = "INTERNAL-AUDIT-KEY-XYZ";
                String companyBankAccount = "ACC1234567890";
                throw new Exception("Revenue cannot be negative: " + declaredRevenue +
                                    ". DB Connection: " + dbConnectionString +
                                    ". Internal Audit Key: " + internalAuditKey +
                                    ". Company Bank Account: " + companyBankAccount);
            }
            // Audit processing logic would go here.
            System.out.println("Audit report processed for financial year " + financialYear + " with revenue: " + declaredRevenue);
        } catch (Exception e) {
            System.err.println("Audit processing error for financial year " + financialYear + ": " + e.getMessage());
            throw new RuntimeException("Failed to process audit for financial year " + financialYear, e);
        }
    }

    public static void main(String[] args) {
        new BAD_FinancialAuditError().processAuditReport("2023", -500000.00);
    }
}
