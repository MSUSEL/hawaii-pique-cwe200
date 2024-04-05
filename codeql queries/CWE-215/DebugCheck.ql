/**
 * @name Sensitive data exposure in debug code
 * @description Identifies instances where sensitive variables are exposed in debug output without proper sanitization, which could lead to sensitive data leaks.
 * @kind problem
 * @problem.severity warning
 * @id CWE-215
 * @tags security
 *      external/cwe/cwe-215
 *      external/cwe/cwe-532
 * @cwe CWE-215
 */

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
    not exists(MethodCall mc |
      mc.getMethod().getName().toLowerCase().matches("%sanitize%") and
      mc.getEnclosingCallable() = this.getEnclosingCallable()
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
