import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_PaymentProcessingServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String paymentId = request.getParameter("paymentId");
        try {
            processPayment(paymentId);
        }catch (ServletException e) {
            response.getWriter().println("Payment processing failed for payment ID: " + paymentId);
        }
    }

    private void processPayment(String paymentId) throws ServletException {
        // Simulation of a failed payment process
        throw new ServletException();
    }
}
