public class BAD_DetailedDatabaseErrorExposure {
    // This example demonstrates an insecure practice where detailed database error messages are directly exposed to the user.

    public static void main(String[] args) {
        try {
            // Attempt to connect to a database or execute a query that might fail
            executeDatabaseQuery("SELECT * FROM sensitive_table");
        } catch (Exception e) {
            // Insecure: Exposing the exception message, potentially containing SQL syntax, table names, or sensitive data.
            System.out.println("Database error: " + e.getMessage());
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for database operation
        throw new Exception("Failed to execute query on 'sensitive_table': Column 'secret_info' does not exist");
    }
}
