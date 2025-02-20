import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_PaymentProcessingServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String cardNumber = request.getParameter("cardNumber");
        String expiryDate = request.getParameter("expiryDate");
        String cvv = request.getParameter("cvv");
        try {
            processPayment(cardNumber, expiryDate, cvv);
        }catch (ServletException e) {
            response.getWriter().println(e.getMessage());
        }
    }

    private void processPayment(String cardNumber, String expiryDate, String cvv) throws ServletException {
        // Simulation of a failed payment process
        throw new ServletException("Payment processing failed for card number: " + cardNumber + " with expiry date: " + expiryDate + " and CVV: " + cvv);  
    }
}
