/**
 * @name CWE-209: Information exposure through a stack trace
 * @description Information from a stack trace propagates to an external user.
 *              Stack traces can unintentionally reveal implementation details
 *              that are useful to an attacker for developing a subsequent exploit.
 * @kind problem
 * @problem.severity error
 * @security-severity 5.4
 * @precision high
 * @id java/stack-trace-exposure/209
 * @tags security
 *       external/cwe/cwe-209
 *       external/cwe/cwe-497
 * @cwe CWE-209
 */

import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.security.StackTraceExposureQuery

from Expr externalExpr, Expr errorInformation
where
  printsStackExternally(externalExpr, errorInformation) or
  stringifiedStackFlowsExternally(DataFlow::exprNode(externalExpr), errorInformation) or
  getMessageFlowsExternally(DataFlow::exprNode(externalExpr), DataFlow::exprNode(errorInformation))
select externalExpr, "$@ can be exposed to an external user.", errorInformation, "Error information"
