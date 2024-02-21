import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets

/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/sensitive-info-leak-servlet
 * @tags security
 *       external/cwe/cwe-536
 */
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
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      // Sinks to the servlet response
      ma.getMethod().hasName("write") and
      ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
      ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      sink.asExpr() = ma.getAnArgument()
    ) or
    exists(MethodAccess ma |
      // Sinks using PrintWriter
      ma.getMethod().hasName("println") and
      ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      sink.asExpr() = ma.getAnArgument()
    )
  }
}

from SensitiveInfoLeakServletConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink, source, sink, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
