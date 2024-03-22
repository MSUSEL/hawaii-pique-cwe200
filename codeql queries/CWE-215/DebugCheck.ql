import java
import semmle.code.java.security.SensitiveVariables

// Identify Debug Flag Checks
class DebugFlagCheck extends IfStmt {
  DebugFlagCheck() {
    exists(VarAccess va |
      va = this.getCondition() and
      va.getVariable().getName().toLowerCase().regexpMatch(".*debug.*")
    )
  }
  
  // Check if a given expression is inside this 'then' branch
  predicate isInThenBranch(Expr e) {
    e.getEnclosingStmt().getParent() = this.getThen()
  }
  
  // Ensure no sanitization method is in the same method scope as the debug check
  predicate noSanitizationInScope() {
    not exists(MethodAccess ma |
      ma.getMethod().getName().toLowerCase().matches("%sanitize%") and
      ma.getEnclosingCallable() = this.getEnclosingCallable()
    )
  }
}

// Query to select sensitive variable usages within the debug flag's 'then' branch,
// excluding those that are sanitized or in scopes with sanitization methods
from DebugFlagCheck dfc, SensitiveVariableExpr sve
where 
  dfc.isInThenBranch(sve) and
  dfc.noSanitizationInScope()
select sve, "Sensitive variable exposed in debug code."
