public class BAD_SalaryDivisionExceptionHandling {
    public static void main(String[] args) {
        int hoursWorked = getConfidentialBaseSalary();
        int totalSalary = getNumberOfEmployees();
        String employeeName = "George";
        String employeeAddress = "123 Main St";

        try {
            if (hoursWorked == 0) {
                throw new ArithmeticException("Cannot divide by zero, employee "+  employeeName + " at " + employeeAddress + " has not worked any hours.");
            } else {
                int hourlyRate = totalSalary / hoursWorked;
                System.out.println("Hourly rate for employee " + employeeName + " is: " + hourlyRate);
            }
        } catch (ArithmeticException e) {
            System.err.println(e.getMessage());
        }
    }

    private static int getConfidentialBaseSalary() {
        return 50000;  // Example: $50,000 total payroll
    }

    private static int getNumberOfEmployees() {
        // Simulate retrieving the number of employees
        return 0;  // Example of a critical input issue, possibly leading to division by zero
    }
}
