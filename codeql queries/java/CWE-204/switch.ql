/**
 * @name CWE-204: Observable discrepancies in sensitive error messages guarded by switch statements
 * @description Detects if statements within sensitive contexts that produce different error messages based on conditional branches, which could lead to observable discrepancies.
 * @kind problem
 * @problem.severity warning
 * @id java/error-message-discrepancies/204
 * @tags security
 *       external/cwe/cwe-204
 * @cwe CWE-204
 */

import java
import SensitiveInfo.SensitiveInfo

/**
 * A class to identify observable discrepancies within switch statements.
 * This includes different method calls, print statements, or potentially logging calls with different arguments.
 */
class ObservableDiscrepancySwitch extends SwitchStmt {
  ObservableDiscrepancySwitch() {
    // Check for different method calls in switch cases
    exists(SwitchCase sc1, SwitchCase sc2, MethodCall ma1, MethodCall ma2 |
      sc1.getEnclosingStmt() = this and
      sc2.getEnclosingStmt() = this and
      sc1 != sc2 and
      ma1.getEnclosingStmt() = sc1 and
      ma2.getEnclosingStmt() = sc2 and
      ma1.getMethod() != ma2.getMethod()
    )
    or
    // Check for print statements with different arguments in switch cases
    exists(SwitchCase sc, ExprStmt es, MethodCall printCall |
      sc.getEnclosingStmt() = this and
      es.getParent() = sc and
      printCall = es.getExpr() and
      // Checks to see if there is a sink in this switch statement
      printCall.getMethod().getName().regexpMatch(getSinkName()) and

      // Basic (Delete later on, after testing getSinkName is working correctly) Match print or println methods from java.io.PrintStream
      printCall.getMethod().getName().regexpMatch("print(ln)?") and
      printCall.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintStream") and
      
      // Check for another print statement in a different case with a different argument
      exists(SwitchCase otherSc, ExprStmt otherEs, MethodCall otherPrintCall |
        otherSc.getEnclosingStmt() = this and
        otherSc != sc and
        otherEs.getParent() = otherSc and
        otherPrintCall = otherEs.getExpr() and
        otherPrintCall.getMethod() = printCall.getMethod() and
        // Ensure the arguments of the print calls are different
        not exists(int i |
          printCall.getArgument(i) = otherPrintCall.getArgument(i)
        )
      )
    )
  }
}

from ObservableDiscrepancySwitch ods
select ods, "This switch statement may lead to observable discrepancies due to different method calls or print statements in cases."