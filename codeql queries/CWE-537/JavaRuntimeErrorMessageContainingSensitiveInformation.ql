/**
 * @name CWE-537: Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/runtime-error-info-exposure/537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<RuntimeSensitiveInfoExposureConfig>;
 
 import Flow::PathGraph
 
 module RuntimeSensitiveInfoExposureConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve) or 
     // Direct access to the exception variable itself
     exists(CatchClause cc | source.asExpr() = cc.getVariable().getAnAccess()) or
     // Consider any method call on the exception object as a source
     exists(CatchClause cc, MethodCall mc | mc.getQualifier() = cc.getVariable().getAnAccess() and source.asExpr() = mc)
   }
 
   predicate isSink(DataFlow::Node sink) {
     // Consider the case where the sink exposes sensitive info within a catch clause of type RuntimeException
     exists(MethodCall mc, CatchClause cc |
       cc.getACaughtType().getASupertype*().hasQualifiedName("java.lang", "RuntimeException") and
       mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
       (
         getSinkAny(sink) or
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrorSink(sink) or
         CommonSinks::isServletSink(sink) or
         getSinkAny(sink)
       ) and
       sink.asExpr() = mc.getAnArgument()
     )
     or
     // Consider the case where the sink is a throw statement that has a super type of RuntimeException
     exists(ThrowStmt ts, ConstructorCall cc |
       // Identifying throw statements creating RuntimeExceptions with sensitive information
       ts.getThrownExceptionType().getASupertype*().hasQualifiedName("java.lang", "RuntimeException") and
       // Throw statements don't have an argument, so you need to look at the ConstructorCall that creates the exception
       cc = ts.getExpr().(ConstructorCall) and
       sink.asExpr() = cc.getAnArgument()
     )
   }
 
   predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%")) and
    // Consider both arguments and the return of sanitization/encryption methods as barriers
    (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
    )
  }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink,
   "CWE-537: Java runtime error message containing sensitive information"
 