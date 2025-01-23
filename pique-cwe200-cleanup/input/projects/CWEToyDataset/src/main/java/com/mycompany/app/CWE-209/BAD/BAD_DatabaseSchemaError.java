public class BAD_DatabaseSchemaError {
    public void updateDatabaseSchema(String sqlUpdateCommand) {
        try {
            if (!sqlUpdateCommand.startsWith("ALTER TABLE")) {
                throw new Exception("Only table alterations are supported." + sqlUpdateCommand);
            }
            // Database schema update logic
        } catch (Exception e) {
            System.err.println("Database schema update failed " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        new BAD_DatabaseSchemaError().updateDatabaseSchema("DROP TABLE Users");
    }
}
