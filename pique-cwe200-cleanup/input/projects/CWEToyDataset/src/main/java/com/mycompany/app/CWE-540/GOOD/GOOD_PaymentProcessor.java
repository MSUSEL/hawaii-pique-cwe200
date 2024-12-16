public class GOOD_PaymentProcessor {
    public void processPayment(String amount) {
        String paymentServiceUrl = System.getenv("PAYMENT_SERVICE_URL");
        String serviceApiKey = System.getenv("SERVICE_API_KEY");

        System.out.println("Processing payment of " + amount + " to " + paymentServiceUrl + " with API key " + serviceApiKey);
        // Actual payment processing logic would be here
    }
}
