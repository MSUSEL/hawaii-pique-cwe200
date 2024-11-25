import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_TransactionServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String transactionId = request.getParameter("transactionId");
        try {
            processTransaction(transactionId);
        } catch (ServletException e) {
            response.getWriter().println("Transaction processing failed for transaction ID: " + transactionId);
        }
    }

    private void processTransaction(String transactionId) throws ServletException {
        // Transaction processing logic can throw exception
        throw new ServletException("Processing error for transaction ID: " + transactionId);
    }
}
