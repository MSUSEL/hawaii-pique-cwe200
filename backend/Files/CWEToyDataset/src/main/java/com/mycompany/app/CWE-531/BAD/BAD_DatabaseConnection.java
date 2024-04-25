import org.junit.Test;
import static org.junit.Assert.assertTrue;

class Database {
    private String dbUrl, dbUser, dbPassword;

    public Database(String dbUrl, String dbUser, String dbPassword) {
        this.dbUrl = dbUrl;
        this.dbUser = dbUser;
        this.dbPassword = dbPassword;
    }

    public boolean connect() {
        return dbUrl.contains("testdb") && dbPassword.equals("password");
    }
}

public class BAD_DatabaseConnection {
    @Test
    public void testConnect() {
        String dbUrl = "jdbc:mysql://localhost/testdb";
        String dbUser = "testuser";
        String dbPassword = "password"; // Hardcoded sensitive password
        Database db = new Database(dbUrl, dbUser, dbPassword);
        assertTrue(db.connect()); // Asserts that the connection is successful
    }
}
