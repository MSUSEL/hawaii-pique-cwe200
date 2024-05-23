import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_TransactionServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String transactionId = request.getParameter("transactionId");
        try {
            processTransaction(transactionId);
        } catch (ServletException e) {
            // Exposing transaction ID in the error message
            response.getWriter().println("Transaction processing failed for ID: " + transactionId + ". Error has been logged.");
            getServletContext().log("Transaction error for ID: " + transactionId, e);
        }
    }

    private void processTransaction(String transactionId) throws ServletException {
        // Transaction processing logic can throw exception
        throw new ServletException("Database timeout during transaction for ID: " + transactionId);
    }
}
