/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/CWE-536
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
    exists(MethodCall mc |
      // Ensuring write is called on servlet response
      mc.getMethod().hasName("write") and
      mc.getQualifier().(MethodCall).getMethod().hasName("getWriter") and
      mc.getQualifier().(MethodCall).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      mc.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = mc.getAnArgument()
    ) 
    or
    exists(MethodCall mc |
      // Inferring println on PrintWriter obtained from servlet response, assuming context
      mc.getMethod().hasName("println") and
      mc.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      mc.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      // Additional context checks might be added here to more directly associate with servlets
      sink.asExpr() = mc.getAnArgument()
    )
    or
    exists(MethodCall log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
      log.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = log.getAnArgument()
   )
    or
    exists(MethodCall log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug"]) and
      log.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = log.getAnArgument()
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
