public class BAD_VulnerableStringComparison {
    public boolean authenticate(String userInput, String secretPassword) {
        return userInput.equals(secretPassword);
    }
}