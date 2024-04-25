import org.junit.Test;
import static org.junit.Assert.assertTrue;
import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

class Database {
    private String dbUrl, dbUser, dbPassword;

    public Database(String dbUrl, String dbUser, String dbPassword) {
        this.dbUrl = dbUrl;
        this.dbUser = dbUser;
        this.dbPassword = dbPassword;
    }

    public boolean connect() {
        // Simulates a condition for connection success
        return !(this.dbPassword.isEmpty() && this.dbUrl.isEmpty() && this.dbUser.isEmpty());
    }
}

public class GOOD_DatabaseConnection {
    @Test
    public void testConnect() {
        Properties props = loadProperties();
        String dbUrl = props.getProperty("dbUrl");
        String dbUser = props.getProperty("dbUser");
        String dbPassword = props.getProperty("dbPassword");

        Database db = new Database(dbUrl, dbUser, dbPassword);
        assertTrue(db.connect()); // Asserts that the connection is successful
    }

    private Properties loadProperties() {
        Properties props = new Properties();
        try (FileInputStream in = new FileInputStream("db_test.properties")) {
            props.load(in);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return props;
    }
}
