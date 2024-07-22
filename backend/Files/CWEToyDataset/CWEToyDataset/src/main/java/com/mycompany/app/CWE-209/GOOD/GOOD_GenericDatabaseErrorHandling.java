public class GOOD_GenericDatabaseErrorHandling {
    public static void main(String[] args) {
        try {
            executeDatabaseQuery("SELECT * FROM users");
        } catch (Exception e) {
            System.out.println("An error occurred while accessing the database. Please try again or contact support.");
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        throw new Exception("Database access error");
    }
}
