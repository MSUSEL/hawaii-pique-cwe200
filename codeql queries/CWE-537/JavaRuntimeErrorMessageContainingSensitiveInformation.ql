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
    //  exists(VarAccess var |
    //    var.getVariable().getName() = "apiKey" and
    //    source.asExpr() = var
    //  )
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Combined logic for detecting sensitive data flowing into exception constructors and linking throw to catch
   predicate isAdditionalFlowStep(DataFlow::Node node1, DataFlow::Node node2) {
    // Flow from sensitive variable to exception in the same method
    exists(ThrowStmt t |
        t.getEnclosingCallable() = node1.asExpr().getEnclosingCallable() and
        t.getEnclosingCallable() = node2.asExpr().getEnclosingCallable() and
        t.getExpr() = node2.asExpr() and
        node1.asExpr() instanceof BinaryExpr // Track sensitive data concatenation
    ) or
    // Flow from throw to catch within the same method
    exists(CatchClause c |
        c.getEnclosingCallable() = node1.asExpr().getEnclosingCallable() and
        c.getEnclosingCallable() = node2.asExpr().getEnclosingCallable() and
        node2.asExpr() = c.getVariable().getAnAccess() and
        exists(ThrowStmt t |
            t.getEnclosingCallable() = c.getEnclosingCallable() and
            node1.asExpr() = t.getExpr()
        )
    ) or
    // Track method A calling method B and propagating the exception back to A
    exists(MethodCall call |
        // Method A calls method B (node1 is the callee)
        call.getCallee() = node1.asExpr().getEnclosingCallable() and
        // Ensure node2 is within the method that calls method B (Method A)
        call.getEnclosingCallable() = node2.asExpr().getEnclosingCallable() and
        // Ensure method B throws the exception
        exists(ThrowStmt t |
            t.getEnclosingCallable() = call.getCallee() and
            node1.asExpr() = t.getExpr()
        )
    ) and
    // Ensure node2 is within Method A’s control flow, handling the exception
    exists(CatchClause catchStmt |
        catchStmt.getEnclosingCallable() = node2.asExpr().getEnclosingCallable() and
        node2.asExpr() = catchStmt.getVariable().getAnAccess()
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
 