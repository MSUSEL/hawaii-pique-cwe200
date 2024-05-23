public class BAD_DatabaseSchemaError {
    public void updateDatabaseSchema(String sqlUpdateCommand) {
        try {
            if (!sqlUpdateCommand.startsWith("ALTER TABLE")) {
                throw new UnsupportedOperationException("Only table alterations are supported.");
            }
            // Database schema update logic
        } catch (UnsupportedOperationException e) {
            System.err.println("Database schema update failed with command: " + sqlUpdateCommand);
        }
    }

    public static void main(String[] args) {
        new BAD_DatabaseSchemaError().updateDatabaseSchema("DROP TABLE Users");
    }
}
