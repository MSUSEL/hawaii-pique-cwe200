import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_TransactionServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String amount = request.getParameter("amount");
        String userEmail = request.getParameter("userId");
        String creditCardNumber = request.getParameter("creditCardNumber");
        try {
            processTransaction(amount, userEmail, creditCardNumber);
        } catch (ServletException e) {
            response.getWriter().println(e.getMessage());
        }
    }

    private void processTransaction(String amount, String userEmail, String creditCardNumber) throws ServletException {
        // Transaction processing logic can throw exception
        throw new ServletException("Transaction processing failed with amount: " + amount + " for user: " + userEmail + " with credit card number: " + creditCardNumber);
    }
}
