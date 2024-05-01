public class GOOD_DatabaseSchemaError {
    public void updateDatabaseSchema(String sqlUpdateCommand) {
        try {
            if (!sqlUpdateCommand.startsWith("ALTER TABLE")) {
                // Generic error message without specific command details
                throw new UnsupportedOperationException("Unsupported operation. Only table alterations are allowed.");
            }
            // Database schema update logic
        } catch (UnsupportedOperationException e) {
            // Log the error generically without exposing the SQL command
            System.err.println("Failed to update the database schema. Please check the command format.");
            // Optionally, rethrow as a more generic exception to avoid leaking implementation details
            throw new RuntimeException("Database update error occurred.");
        }
    }

    public static void main(String[] args) {
        new GOOD_DatabaseSchemaError().updateDatabaseSchema("ALTER TABLE Users ADD COLUMN Age INT");
    }
}
