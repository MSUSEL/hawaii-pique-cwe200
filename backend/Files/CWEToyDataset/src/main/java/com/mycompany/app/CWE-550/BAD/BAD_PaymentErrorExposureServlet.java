import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_PaymentErrorExposureServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String creditCardNumber = request.getParameter("creditCardNumber");
        String amount = request.getParameter("amount");
        try {
            processPayment(amount, creditCardNumber);
        } catch (Exception e) {
            // Insecure: Exposing payment processing exception details to the client
            response.getWriter().write("Payment processing failed: " + creditCardNumber + amount + e.getMessage());
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }

    private void processPayment(String creditCardNumber, String amount) throws Exception {
        if (creditCardNumber == null || amount == null) {
            throw new Exception("Invalid payment data provided." + creditCardNumber + amount);
        }
        // Simulate a payment gateway failure
        throw new Exception("Payment gateway timeout during transaction." + creditCardNumber + amount);
    }
}
