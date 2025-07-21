/**
 * @name CWE-215: Sensitive data exposure in debug code
 * @description Identifies instances where sensitive variables are exposed in debug output without proper sanitization, which could lead to sensitive data leaks.
 * @kind path-problem
 * @problem.severity warning
 * @id java/debug-code-sensitive-data-exposure/215
 * @tags security
 *      external/cwe/cwe-215
 * @cwe CWE-215
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 module Flow = TaintTracking::Global<SensitiveDataConfig>;
 import Flow::PathGraph

 
 // Define sensitive data sources (e.g., sensitive variables)
 module SensitiveDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Define debug output sinks (e.g., print statements or logging methods)
  predicate isSink(DataFlow::Node sink) {
      getSinkAny(sink)
   }

   predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
  } 
 }
 
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
 
 // Query for sensitive data flow within debug flag's 'then' branch
 from Flow::PathNode source, Flow::PathNode sink, DebugFlagCheck dfc
 where
   Flow::flowPath(source, sink) and       // Data flow path exists from source to sink
   dfc.isInThenBranch(sink.getNode().asExpr()) and  // Sink is within debug flag’s 'then' branch
   dfc.noSanitizationInScope()             // No sanitization in scope
 select sink.getNode(), source, sink, "Sensitive data exposed in debug code."
 