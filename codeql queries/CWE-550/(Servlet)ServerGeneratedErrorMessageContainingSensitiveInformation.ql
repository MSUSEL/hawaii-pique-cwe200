/**
 * @name CWE-550: Exposure of sensitive information through servlet responses
 * @description Detects when sensitive information from exceptions or system details
 *              is exposed to clients via servlet responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */

import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<HttpServletExceptionSourceConfig>;

import Flow::PathGraph

// Defines a configuration for tracking the flow of sensitive information in HttpServlets
module HttpServletExceptionSourceConfig implements DataFlow::ConfigSig {
  // Identifies sources of sensitive information within servlet methods
  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc, Method m, CatchClause cc |
      // Ensures the method is part of a class that extends HttpServlet
      m.getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      (
        (
          // Direct access to the exception variable itself
          source.asExpr() = cc.getVariable().getAnAccess()
          or
          // Consider any method call on the exception object as a source
          mc.getQualifier() = cc.getVariable().getAnAccess() and source.asExpr() = mc
        ) and
        source.asExpr() = mc and
        // The call must occur within the servlet method
        mc.getEnclosingCallable() = m
      )
      or
      exists(SensitiveVariableExpr sve |
        source.asExpr() = sve and
        // The call must occur within the servlet method
        sve.getEnclosingCallable() = m
      )
    )
  }

  // Defines sinks where sensitive information could be exposed to clients
  predicate isSink(DataFlow::Node sink) {

    exists(CatchClause cc, MethodCall mc |
      // Ensure the MethodCall is within the CatchClause
      mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
      // Ensure the sink matches one of the known sensitive sinks
      (
        CommonSinks::isErrPrintSink(sink) or
        CommonSinks::isServletSink(sink) or
        getSinkAny(sink)
      ) and
      // Link the sink to the argument of the MethodCall
      sink.asExpr() = mc.getAnArgument()
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


// Executes the configuration to find data flows from identified sources to sinks
from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "CWE-550: (Servlet) Server-Generated Error Message Containing Sensitive Information."
