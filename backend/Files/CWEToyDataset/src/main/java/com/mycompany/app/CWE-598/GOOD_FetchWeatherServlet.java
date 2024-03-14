import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;

@WebServlet("/getWeather")
public class GOOD_FetchWeatherServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String city = request.getParameter("city");
        
        // Call to a weather API or internal logic to fetch weather data for the specified city
        // The city parameter is not considered sensitive, making it suitable for a GET request.
        
        response.setContentType("application/json");
        response.getWriter().write("{\"weather\":\"Sunny\", \"city\":\"" + city + "\"}");
    }
}
