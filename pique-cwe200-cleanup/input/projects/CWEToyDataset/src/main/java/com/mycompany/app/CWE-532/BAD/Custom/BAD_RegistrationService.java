public class BAD_RegistrationService {
    public void registerUser(String name, String email, String ssn) {
        EventChronicle.logEvent("New user registered: Name - " + name + ", Email - " + email + ", SSN - " + ssn);
    }
}
