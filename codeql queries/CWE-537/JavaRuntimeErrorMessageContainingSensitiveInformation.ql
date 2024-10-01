/**
 * @name CWE-537: Java Runtime Error Message Containing Sensitive Information
 * @description Detects sensitive information (e.g., apiKey) being added to an exception message and then exposed via getMessage in HTTP responses.
 * @kind path-problem
 * @problem.severity error
 * @id java/runtime-error-message-exposure/537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.DataFlow
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<SensitiveInfoInExceptionConfig>;

import Flow::PathGraph

/**
 * Defines the taint configuration for tracking sensitive data in exception messages.
 */
module SensitiveInfoInExceptionConfig implements DataFlow::ConfigSig {

  // Sensitive variables (e.g., apiKey)
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
  }

  // Combined logic for detecting sensitive data flowing into exception constructors and linking throw to catch
  predicate isAdditionalFlowStep(DataFlow::Node node1, DataFlow::Node node2) {
    // Track flow from a throw statement to a catch block (ignore throws inside the catch block itself)
    exists(ThrowStmt t |
        // Ensure the throw statement is outside any catch block
        not exists(CatchClause catchInside |
            // Check if the throw statement is inside the body (block) of the catch clause
            t = catchInside.getBlock().getAStmt()
        ) and
        // Ensure that node1 is the thrown exception (e.g., new Exception(...))
        t.getExpr() = node1.asExpr() and
        // Ensure that both the throw and the catch block are in the same method or constructor
        t.getEnclosingCallable() = node1.asExpr().getEnclosingCallable() and
        t.getEnclosingCallable() = node2.asExpr().getEnclosingCallable() and
        // Ensure the thrown exception is a RuntimeException or its subclass
        t.getExpr().getType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
        // Ensure the exception is caught by a catch block
        exists(CatchClause cc |
            cc.getEnclosingCallable() = t.getEnclosingCallable() and
            // Check that the caught exception is accessed (used in getMessage() or similar)
            cc.getVariable().getAnAccess() = node2.asExpr()
        ) 
    )

  }

  // Sink: Tracks `e.getMessage()` where sensitive information may be exposed
  predicate isSink(DataFlow::Node sink) {
    exists(MethodCall mc |
      mc.getMethod().getName() = "getMessage" and
      mc.getQualifier() instanceof VarAccess and
      // Ensure it's the caught exception variable
      mc.getQualifier().(VarAccess).getVariable() = sink.asExpr().(VarAccess).getVariable()
    )
  }
}

// Perform taint tracking from source to sink
from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "Sensitive information might be exposed through an exception message."
