import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_CryptoSeedExposure extends HttpServlet {

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String userId = request.getParameter("userId");
        CryptoConfig cryptoConfig = getCryptoConfig(userId);
        
        response.setHeader("X-Crypto-Seed", cryptoConfig.getSeedData());
        response.setContentType("application/json");
        response.getWriter().write(encryptData("{\"success\": true, \"content\": \"Your data has been securely stored.\"}", cryptoConfig));
        response.getWriter().close();
    }

    private String encryptData(String data, CryptoConfig cryptoConfig) {
        // Simulate encryption (for demonstration purposes, actual encryption details omitted)
        return "EncryptedData{" + data + "}";
    }

    private CryptoConfig getCryptoConfig(String userId) {
        // Simulate fetching cryptographic configuration for the user
        return new CryptoConfig("secureRandomSeed12345");
    }

    private static class CryptoConfig {
        private String seedData;

        public CryptoConfig(String seedData) {
            this.seedData = seedData;
        }

        public String getSeedData() {
            return seedData;
        }
    }
}
