import javax.crypto.Mac;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.security.MessageDigest;
import org.apache.http.HttpRequest;

public class BAD_MacComparison {
    // Vulnerability: The system uses a timing-unsafe comparison method to validate a message authentication code (MAC),
    // allowing an attacker to exploit timing discrepancies to infer the MAC value.
    // The system should use a constant-time comparison method to prevent timing attacks.
    public boolean validate(HttpRequest request, SecretKey key) throws Exception {
        byte[] message = getMessageFrom(request);
        byte[] signature = getSignatureFrom(request);

        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(key.getEncoded(), "HmacSHA256"));
        byte[] actual = mac.doFinal(message);
        return MessageDigest.isEqual(signature, actual);
    }

    public byte[] getMessageFrom(HttpRequest request) {
        // Assuming the message is stored as a Base64-encoded header named "Message"
        String messageHeader = request.getFirstHeader("Message").getValue();
    
        // Decode the Base64-encoded message
        return java.util.Base64.getDecoder().decode(messageHeader);
    }

    public byte[] getSignatureFrom(HttpRequest request) {
        // Assuming the signature is stored as a Base64-encoded header named "Signature"
        String signatureHeader = request.getFirstHeader("Signature").getValue();
    
        // Decode the Base64-encoded signature
        return java.util.Base64.getDecoder().decode(signatureHeader);
    }
}
