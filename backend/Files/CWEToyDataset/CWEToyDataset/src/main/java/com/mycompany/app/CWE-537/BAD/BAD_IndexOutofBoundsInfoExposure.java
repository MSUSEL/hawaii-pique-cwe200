public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args) {
        String userData = "User:JohnDoe | Password: SuperSecret123! | SSN: 123-45-6789 | Card: 4111-1111-1111-1111";
        String[] dataMap = new String[3]; 

        int index = 5;

        try {
            if (index < 0 || index >= dataMap.length) {
                throw new ArrayIndexOutOfBoundsException("Error: Could not store data at index " + index + ". Attempted value: " + userData);
            }
            else{
                dataMap[index] = userData;
            }
            System.out.println("Data stored successfully at index " + index);
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println(e.getMessage());
        }
    }
}
