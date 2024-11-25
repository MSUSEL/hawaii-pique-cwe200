public class BAD_SalaryDivisionExceptionHandling {
    public static void main(String[] args) {
        int hoursWorked = getConfidentialBaseSalary();
        int totalSalary = getNumberOfEmployees();
        String employeeID = getEmployeeID();

        try {
            if (hoursWorked == 0) {
                throw new ArithmeticException();
            } else {
                int hourlyRate = totalSalary / hoursWorked;
                System.out.println("Hourly rate for employee " + employeeID + " is: " + hourlyRate);
            }
        } catch (ArithmeticException e) {
            System.err.println("Attempt to divide by zero - confidential payroll calculation cannot be completed for employee: " + employeeID);
        }
    }

    private static int getConfidentialBaseSalary() {
        return 50000;  // Example: $50,000 total payroll
    }

    private static int getNumberOfEmployees() {
        // Simulate retrieving the number of employees
        return 0;  // Example of a critical input issue, possibly leading to division by zero
    }

    private static String getEmployeeID() {
        // Simulate retrieving the number of employees
        return "329";  // Example of a critical input issue, possibly leading to division by zero
    }
}
