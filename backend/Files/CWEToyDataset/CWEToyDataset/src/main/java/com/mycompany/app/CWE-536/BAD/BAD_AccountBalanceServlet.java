import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_AccountBalanceServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String accountId = request.getParameter("accountId");
        retrieveAccountBalance(accountId);
    }

    private void retrieveAccountBalance(String accountId) throws ServletException {
        // Supposed database call that fails
        throw new ServletException("Query failed for account: " + accountId);
    }
}
