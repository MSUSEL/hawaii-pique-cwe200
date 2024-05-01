public class GOOD_DetailedDatabaseErrorExposure {
    // This example demonstrates a secure practice where generic error messages are used to avoid information leakage.

    public static void main(String[] args) {
        try {
            // Attempt to connect to a database or execute a query
            executeDatabaseQuery("SELECT * FROM sensitive_table");
        } catch (Exception e) {
            // Secure: Using a generic error message that does not expose any details of the underlying data structure or SQL.
            System.out.println("A database error occurred. Please try again later.");
            // Log the specific error internally where it cannot be accessed by the end-user
            logInternalError("Database query failed: " + e.getMessage());
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for a database operation that might throw an error
        throw new Exception("Query failed due to schema mismatch.");
    }

    private static void logInternalError(String errorDetails) {
        // Internal logging mechanism that users do not have access to
        System.err.println(errorDetails);
    }
}
