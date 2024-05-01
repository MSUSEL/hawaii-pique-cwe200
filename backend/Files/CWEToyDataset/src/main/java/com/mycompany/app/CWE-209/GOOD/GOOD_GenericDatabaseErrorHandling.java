public class GOOD_GenericDatabaseErrorHandling {
    // Illustrates a secure practice by providing generic error messages for database errors, avoiding exposure of sensitive details.

    public static void main(String[] args) {
        try {
            executeDatabaseQuery("SELECT * FROM users");
        } catch (Exception e) {
            // Secure: Providing a generic error message without exposing sensitive information.
            System.out.println("An error occurred while accessing the database. Please try again or contact support.");
        }
    }

    private static void executeDatabaseQuery(String query) throws Exception {
        // Placeholder for database operation that could fail
        throw new Exception("Database access error");
    }
}
