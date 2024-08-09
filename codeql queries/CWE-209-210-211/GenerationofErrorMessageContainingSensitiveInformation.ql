/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/error-message-sensitive-info/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

 import java
 private import semmle.code.java.dataflow.ExternalFlow
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<SensitiveInfoInErrorMsgConfig>;
 import Flow::PathGraph
 /** A configuration for tracking sensitive information flow into error messages. */
 module SensitiveInfoInErrorMsgConfig implements DataFlow::ConfigSig{
   predicate isSource(DataFlow::Node source) {
    // Broad definition, consider refining
    exists(SensitiveVariableExpr sve | source.asExpr() = sve) or 
    // Direct access to the exception variable itself
    exists(CatchClause cc | source.asExpr() = cc.getVariable().getAnAccess()) or
    // Consider any method call on the exception object as a source
    exists(CatchClause cc, MethodCall mc | mc.getQualifier() = cc.getVariable().getAnAccess() and source.asExpr() = mc)
   }

   predicate isSink(DataFlow::Node sink) {
     // Identifying common error message generation points
     CommonSinks::isErrPrintSink(sink) or 
     CommonSinks::isErrorSink(sink) or
     getSinkAny(sink)

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
 select sink.getNode(), source, sink, "CWE-209: Error message may contain sensitive information."
 