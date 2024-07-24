/**
 * @name CWE-536: Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-runtime-error-message-exposure/536
 * @tags security
 *       external/cwe/cwe-536
 * @cwe CWE-536
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.DataFlow
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;

import Flow::PathGraph

module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve | source.asExpr() = sve) or
    // Direct access to the exception variable itself
    exists(CatchClause cc | source.asExpr() = cc.getVariable().getAnAccess()) or
    // Consider any method call on the exception object as a source
    exists(CatchClause cc, MethodCall mc | mc.getQualifier() = cc.getVariable().getAnAccess() and source.asExpr() = mc)
  }

  predicate isSink(DataFlow::Node sink) {
    // Consider the case where the sink exposes sensitive info within a catch clause of type ServletException
    exists(CatchClause cc, MethodCall mc |
      // Ensure the CatchClause is catching ServletException
      cc.getACaughtType().hasQualifiedName("javax.servlet", "ServletException") and
      // Ensure the MethodCall is within the CatchClause for the ServletException
      mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
      // Ensure the sink matches one of the known sensitive sinks
      (
        getSinkAny(sink) or
        //  CommonSinks::isLoggingSink(sink) or
        CommonSinks::isPrintSink(sink) or
        CommonSinks::isServletSink(sink) or
        CommonSinks::isErrorSink(sink) or
        CommonSinks::isIOSink(sink) or
        getSinkAny(sink)
      ) and
      // Link the sink to the argument of the MethodCall
      sink.asExpr() = mc.getAnArgument()
    )
    or
    // Consider the case where the sink is a throw statement that throws a ServletException
    exists(ThrowStmt ts, ConstructorCall cc |
      // Identifying throw statements creating ServletException with sensitive information
      ts.getThrownExceptionType().hasQualifiedName("javax.servlet", "ServletException") and
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
  "CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
