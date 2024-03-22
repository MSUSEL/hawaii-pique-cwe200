/**
 * @name Exposure of sensitive information through servlet responses
 * @description Detects when sensitive information from exceptions or system details
 *              is exposed to clients via servlet responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/http-servlet-sensitive-info-exposure
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */


import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import DataFlow::PathGraph


// Defines a configuration for tracking the flow of sensitive information in HttpServlets
class HttpServletExceptionSourceConfig extends TaintTracking::Configuration {
  HttpServletExceptionSourceConfig() { this = "HttpServletExceptionSourceConfig" }

  // Identifies sources of sensitive information within servlet methods
  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma, Method m |
      // Ensures the method is part of a class that extends HttpServlet
      m.getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      (
        // Captures method calls on Throwable instances that may leak information
        (ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
         ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace", "toString"])) or
        // Includes methods that expose file system paths
        (ma.getMethod().hasName("getAbsolutePath") and
         ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File")) or
        // Considers environment variables and system properties as potential sources
        (ma.getMethod().hasName(["getenv", "getProperty"]) and
         ma.getMethod().getDeclaringType().hasQualifiedName("java.lang", "System")) or
        // Include user input as a potential source
        (ma.getMethod().getDeclaringType().hasQualifiedName("javax.servlet", "ServletRequest") and
        ma.getMethod().hasName(["getParameter", "getAttribute"]))
      ) and
      // The call must occur within the servlet method
      ma.getEnclosingCallable() = m and
      source.asExpr() = ma
    )
  }

  // Defines sinks where sensitive information could be exposed to clients
  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      // Targets servlet response writing methods
      ma.getMethod().hasName("write") and
      // Checks the write method is called on a PrintWriter obtained from HttpServletResponse
      ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
      ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      sink.asExpr() = ma.getAnArgument()
    ) or
    exists(MethodAccess ma |
      // Targets PrintWriter methods that may leak information
      ma.getMethod().hasName("println") and
      ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      sink.asExpr() = ma.getAnArgument()
    ) or 
    exists(MethodAccess ma |
      // Includes HttpServletResponse methods that can expose information
      (ma.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "sendError") or
       ma.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "addHeader") or
       ma.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "setStatus")) and
      sink.asExpr() = ma.getAnArgument()
    ) or
    exists(MethodAccess logMa |
      // Adds logging methods as sinks, considering them potential leak points
      logMa.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      logMa.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = logMa.getAnArgument()
    )
  }
}

// Executes the configuration to find data flows from identified sources to sinks
from HttpServletExceptionSourceConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive information from an exception might be exposed to clients."
