import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class GOOD_AccountBalanceServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String accountId = request.getParameter("accountId");
        try {
            retrieveAccountBalance(accountId);
        } catch (ServletException e) {
            response.getWriter().println("Unable to retrieve balance.");
        }
    }

    private void retrieveAccountBalance(String accountId) throws ServletException {
        // Supposed database call that fails
        throw new ServletException("Query failed.");
    }
}
