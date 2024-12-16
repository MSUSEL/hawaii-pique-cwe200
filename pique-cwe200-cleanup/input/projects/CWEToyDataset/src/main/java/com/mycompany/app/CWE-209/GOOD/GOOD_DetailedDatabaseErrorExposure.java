public class GOOD_DetailedDatabaseErrorExposure {

    public static void main(String[] args) {
        try {
            // Attempt to connect to a database or execute a query
            executeDatabaseQuery("SELECT * FROM sensitive_table");
        } catch (Exception e) {
            System.out.println("A database error occurred. Please try again later.");
            displayInternalError("Database query failed.");
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for a database operation that might throw an error
        throw new Exception("Query failed due to schema mismatch.");
    }

    private static void displayInternalError(String errorDetails) {
        System.err.println(errorDetails);
    }
}
