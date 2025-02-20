import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_AccountBalanceServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String accountId = request.getParameter("accountId");
        try {
            retrieveAccountBalance(accountId);
        } catch (ServletException e) {
            response.getWriter().println(e.getMessage());
        }
    }

    private void retrieveAccountBalance(String accountId) throws ServletException {
        // Supposed database call that fails
        throw new ServletException("Query failed for account: " + accountId);
    }
}
