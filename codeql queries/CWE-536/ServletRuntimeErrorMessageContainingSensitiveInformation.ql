/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/536
 * @tags security
 *       external/cwe/cwe-536
 * @cwe CWE-536
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.security.SensitiveVariables
import CommonSinks.CommonSinks

module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;
import Flow::PathGraph

module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {

  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      // Sources from exceptions
      mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      (mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
      source.asExpr() = mc
    )
    or
    exists(MethodCall mc |
      // Additional sources: Sensitive file paths
      mc.getMethod().hasName("getAbsolutePath") and
      mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
      source.asExpr() = mc
    )
    or
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }

  predicate isSink(DataFlow::Node sink) {
    // Ensure that all sinks are within servlets
    exists(MethodCall mc | sink.asExpr() = mc.getAnArgument() and
      mc.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet")
    ) and  
    (CommonSinks::isServletSink(sink) or
    CommonSinks::isPrintSink(sink) or
    CommonSinks::isLoggingSink(sink)) or 
    CommonSinks::isErrorSink(sink)
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
