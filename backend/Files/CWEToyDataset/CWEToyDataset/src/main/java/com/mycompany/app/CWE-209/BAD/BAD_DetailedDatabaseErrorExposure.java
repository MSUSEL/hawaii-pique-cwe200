public class BAD_DetailedDatabaseErrorExposure {
    public static void main(String[] args) {
        try {
            executeDatabaseQuery("SELECT * FROM sensitive_table");
        } catch (Exception e) {
            System.out.println("Database error: " + e.getMessage());
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for database operation
        throw new Exception("Failed to execute query on 'sensitive_table': Column 'secret_info' does not exist");
    }
}
