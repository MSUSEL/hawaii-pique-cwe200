import org.junit.Test;
import static org.junit.Assert.assertEquals;

class PersonalDataHandler {
    public String handlePersonalData(String personalData) {
        return "Handled " + personalData;
    }
}

public class BAD_PersonalDataHandler {
    @Test
    public void testHandlePersonalData() {
        String personalData = "SSN: 123-45-6789"; // Exposing sensitive personal information
        PersonalDataHandler handler = new PersonalDataHandler();
        assertEquals("Handled SSN: 123-45-6789", handler.handlePersonalData(personalData));
    }
}
