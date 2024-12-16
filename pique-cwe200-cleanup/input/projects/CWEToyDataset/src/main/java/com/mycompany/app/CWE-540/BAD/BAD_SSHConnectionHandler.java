import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;

public class BAD_SSHConnectionHandler {
    private static final String SSH_USER = "admin";
    private static final String SSH_PASSWORD = "adminPass";

    public Session createSSHSession(String host) throws Exception {
        JSch jsch = new JSch();
        Session session = jsch.getSession(SSH_USER, host, 22);
        session.setPassword(SSH_PASSWORD);
        session.setConfig("StrictHostKeyChecking", "no");
        session.connect();
        return session;
    }
}
