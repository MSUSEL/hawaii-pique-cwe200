public class BAD_DetailedDatabaseErrorExposure {
    public static void main(String[] args) {
        String query = "SELECT * FROM accounts_table";
        try {
            executeDatabaseQuery(query);
        } catch (Exception e) {
            System.out.println("Database error: Failed to execute query: " + query);
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for database operation
        throw new Exception();
    }
}
