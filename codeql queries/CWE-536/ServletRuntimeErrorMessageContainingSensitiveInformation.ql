/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-536
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
import DataFlow::PathGraph


class SensitiveInfoLeakServletConfig extends TaintTracking::Configuration {
  SensitiveInfoLeakServletConfig() { this = "SensitiveInfoLeakServletConfig" }


  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      // Sources from exceptions
      ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
      source.asExpr() = ma
    )
    or
    exists(MethodAccess ma |
      // Additional sources: Sensitive file paths
      ma.getMethod().hasName("getAbsolutePath") and
      ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
      source.asExpr() = ma
    )
    or
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      // Ensuring write is called on servlet response
      ma.getMethod().hasName("write") and
      ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
      ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      ma.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = ma.getAnArgument()
    ) 
    or
    exists(MethodAccess ma |
      // Inferring println on PrintWriter obtained from servlet response, assuming context
      ma.getMethod().hasName("println") and
      ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      ma.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      // Additional context checks might be added here to more directly associate with servlets
      sink.asExpr() = ma.getAnArgument()
    )
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
      log.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = log.getAnArgument()
   )
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug"]) and
      log.getEnclosingCallable().getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      sink.asExpr() = log.getAnArgument()
    )
  }
}

from SensitiveInfoLeakServletConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
