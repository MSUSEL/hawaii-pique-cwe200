import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.net.SocketTimeoutException;

public class BAD_PaymentErrorExposureServlet extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String creditCardNumber = request.getParameter("creditCardNumber");
        String amount = request.getParameter("amount");
        try {
            processPayment(amount, creditCardNumber);
        } catch (SocketTimeoutException e) {
            response.getWriter().write("Payment processing failed for card " + creditCardNumber +
                                       " and amount $" + amount + ": " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }

    private void processPayment(String amount, String creditCardNumber) throws SocketTimeoutException {
        if (creditCardNumber == null || amount == null) {
            throw new SocketTimeoutException("Invalid payment data provided for card " + creditCardNumber +
                                " and amount $" + amount);
        }
        throw new SocketTimeoutException("Payment gateway timeout during transaction for card " + creditCardNumber +
                            " and amount $" + amount);
    }
}