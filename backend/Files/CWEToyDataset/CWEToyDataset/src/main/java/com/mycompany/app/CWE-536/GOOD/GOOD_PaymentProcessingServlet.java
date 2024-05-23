import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class GOOD_PaymentProcessingServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String paymentId = request.getParameter("paymentId");
        try {
            processPayment(paymentId);
        } catch (ServletException e) {
            // Handling the exception by exposing payment ID
            response.getWriter().println("Payment processing failed.");
        }
    }

    private void processPayment(String paymentId) throws ServletException {
        // Simulation of a failed payment process
        throw new ServletException("Transaction error.");
    }
}
