/**
 * @name CWE-215: Sensitive data exposure in debug code
 * @description Identifies instances where sensitive variables are exposed in debug output without proper sanitization, which could lead to sensitive data leaks.
 * @kind path-problem
 * @problem.severity warning
 * @id python/debug-code-sensitive-data-exposure/215
 * @tags security
 *      external/cwe/cwe-215
 * @cwe CWE-215
 */

 import python
 import semmle.python.dataflow.new.TaintTracking
 import semmle.python.dataflow.new.DataFlow
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 module Flow = TaintTracking::Global<SensitiveDataConfig>;
 import Flow::PathGraph

 
 // Define sensitive data sources (e.g., sensitive variables)
 module SensitiveDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Define debug output sinks (e.g., print statements or logging methods)
  predicate isSink(DataFlow::Node sink) {
      CommonSinks::isPrintSink(sink) or
      CommonSinks::isErrPrintSink(sink) or
      getSinkAny(sink)
   }

   predicate isBarrier(DataFlow::Node node) {
    // Add Python-specific barriers if needed
    none()
  } 
 }
 
 // Identify Debug Flag Checks
 class DebugFlagCheck extends If {
   DebugFlagCheck() {
     exists(Name name |
       name = this.getTest() and
       name.getId().toLowerCase().regexpMatch(".*debug.*")
     ) or
     exists(Compare cmp |
       cmp = this.getTest() and
       exists(Name name |
         name = cmp.getLeft() and
         name.getId().toLowerCase().regexpMatch(".*debug.*")
       )
     ) or
     exists(Attribute attr |
       attr = this.getTest() and
       attr.getName().toLowerCase().regexpMatch(".*debug.*")
     )
   }
   
   // Check if a given expression is inside this 'then' branch (body)
   predicate isInThenBranch(Expr e) {
     exists(Stmt stmt |
       stmt = this.getBody().getAnItem() and
       e.getParent*() = stmt
     )
   }
   
   // Ensure no sanitization method is in the same function scope as the debug check
   predicate noSanitizationInScope() {
     not exists(Call call |
       call.getFunc().(Name).getId().toLowerCase().matches("%sanitize%") and
       call.getScope() = this.getScope()
     ) and
     not exists(Call call |
       call.getFunc().(Attribute).getName().toLowerCase().matches("%sanitize%") and
       call.getScope() = this.getScope()
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
 