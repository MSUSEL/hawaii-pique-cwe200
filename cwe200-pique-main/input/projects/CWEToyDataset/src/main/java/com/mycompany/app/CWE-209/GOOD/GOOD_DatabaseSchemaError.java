public class GOOD_DatabaseSchemaError {
    public void updateDatabaseSchema(String sqlUpdateCommand) {
        try {
            if (!sqlUpdateCommand.startsWith("ALTER TABLE")) {
                throw new UnsupportedOperationException("Unsupported operation. Only table alterations are allowed.");
            }
            // Database schema update logic
        } catch (UnsupportedOperationException e) {
            System.err.println("Failed to update the database schema. Please check the command format.");
            throw new RuntimeException("Database update error occurred.");
        }
    }

    public static void main(String[] args) {
        new GOOD_DatabaseSchemaError().updateDatabaseSchema("ALTER TABLE Users ADD COLUMN Age INT");
    }
}
