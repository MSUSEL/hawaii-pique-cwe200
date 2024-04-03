import java

// Helper predicate to check if a term is considered suspicious based on the file path
predicate isTermSuspiciousForFile(string term, string filePath) {
    (filePath.matches(".*sensitive/config/.*") and (term = "password" or term = "username"))
}

class SensitiveVariableExpr2 extends Expr {
    SensitiveVariableExpr2() {
        exists(Variable v, string filePath |
            this = v.getAnAccess() and
            not this instanceof CompileTimeConstantExpr and
            // Use the location of this variable access to correctly get the file path
            filePath = this.getLocation().getFile().getAbsolutePath() and

            // Check if the variable name is suspicious within the context of its file location
            isTermSuspiciousForFile(v.getName(), filePath)
        )
    }
}

from SensitiveVariableExpr2 sve
select sve, sve.getLocation().getFile().getAbsolutePath(), "Suspicious variable access found."
