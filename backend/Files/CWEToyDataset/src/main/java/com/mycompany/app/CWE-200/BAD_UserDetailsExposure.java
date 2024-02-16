import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class BAD_UserDetailsExposure extends HttpServlet {
    // This servlet demonstrates an insecure practice of including sensitive user details directly in web application responses.

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) {
        String userId = request.getParameter("userId");

        try {
            // Simulate fetching user details based on the provided userId
            String userDetails = getUserDetails(userId);
            // Insecure: Directly sending sensitive user details in the response
            response.getWriter().write("User details: " + userDetails);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private String getUserDetails(String userId) {
        // Placeholder method to simulate fetching sensitive user details
        // In a real application, this might include personal information, such as email, address, etc.
        return "Name: John Doe, Email: john.doe@example.com, Phone: 555-0100";
    }
}
