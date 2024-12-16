import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_AccountBalanceServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String accountId = request.getParameter("accountId");
        try {
            retrieveAccountBalance(accountId);
        } catch (ServletException e) {
            // Directly using account ID from the exception's context
            response.getWriter().println("Unable to retrieve balance for account ID: " + accountId);
        }
    }

    private void retrieveAccountBalance(String accountId) throws ServletException {
        // Supposed database call that fails
        throw new ServletException("Query failed for account: " + accountId);
    }
}
