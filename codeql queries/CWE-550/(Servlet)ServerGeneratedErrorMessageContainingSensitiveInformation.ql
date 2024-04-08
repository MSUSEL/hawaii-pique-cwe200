/**
 * @name Exposure of sensitive information through servlet responses
 * @description Detects when sensitive information from exceptions or system details
 *              is exposed to clients via servlet responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/CWE-550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */


import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking

module Flow = TaintTracking::Global<HttpServletExceptionSourceConfig>;
import Flow::PathGraph

// Defines a configuration for tracking the flow of sensitive information in HttpServlets
module HttpServletExceptionSourceConfig implements DataFlow::ConfigSig{

  // Identifies sources of sensitive information within servlet methods
  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc, Method m |
      // Ensures the method is part of a class that extends HttpServlet
      m.getDeclaringType().getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
      (
        // Captures method calls on Throwable instances that may leak information
        (mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
        mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace", "toString"])) or
        // Includes methods that expose file system paths
        (mc.getMethod().hasName("getAbsolutePath") and
        mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "File")) or
        // Considers environment variables and system properties as potential sources
        (mc.getMethod().hasName(["getenv", "getProperty"]) and
        mc.getMethod().getDeclaringType().hasQualifiedName("java.lang", "System")) or
        // Include user input as a potential source
        (mc.getMethod().getDeclaringType().hasQualifiedName("javax.servlet", "ServletRequest") and
        mc.getMethod().hasName(["getParameter", "getAttribute"]))
      ) and
      // The call must occur within the servlet method
      mc.getEnclosingCallable() = m and
      source.asExpr() = mc
    )
  }

  // Defines sinks where sensitive information could be exposed to clients
  predicate isSink(DataFlow::Node sink) {
    exists(MethodCall mc |
      // Targets servlet response writing methods
      mc.getMethod().hasName("write") and
      // Checks the write method is called on a PrintWriter obtained from HttpServletResponse
      mc.getQualifier().(MethodCall).getMethod().hasName("getWriter") and
      mc.getQualifier().(MethodCall).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      sink.asExpr() = mc.getAnArgument()
    ) or
    exists(MethodCall mc |
      // Targets PrintWriter methods that may leak information
      mc.getMethod().hasName("println") and
      mc.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      sink.asExpr() = mc.getAnArgument()
    ) or 
    exists(MethodCall mc |
      // Includes HttpServletResponse methods that can expose information
      (mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "sendError") or
      mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "addHeader") or
      mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "setStatus")) and
      sink.asExpr() = mc.getAnArgument()
    ) or
    exists(MethodCall logMa |
      // Adds logging methods as sinks, considering them potential leak points
      logMa.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      logMa.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = logMa.getAnArgument()
    )
  }
}

// Executes the configuration to find data flows from identified sources to sinks
from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-550: (Servlet) Server-Generated Error Message Containing Sensitive Information."
