public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args) {
        String[] dataMap = {"MistyMountain2020", "medicalRecords"};
        int index = 2;

        try {
            if (index < 0 || index >= dataMap.length) {
                throw new IndexOutOfBoundsException();
            } else {
                String data = dataMap[index];
                System.out.println("Data: " + data);
            }
        } catch (IndexOutOfBoundsException e) {
            System.err.println("Index " + index + " is out of bounds for the array" + dataMap[index]);
        }
    }
}
