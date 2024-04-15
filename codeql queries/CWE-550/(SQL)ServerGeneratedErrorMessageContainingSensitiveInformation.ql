/**
 * @name Exposure of sensitive SQL exception information
 * @description Detects potential exposure of sensitive information from SQLExceptions through console output, logging frameworks, or web application responses.
 * @kind path-problem
 * @problem.severity warning
 * @id java/sql-info-exposure/550
 * @tags security
 *       external/cwe/cwe-209
 *       external/cwe/cwe-215
 *       external/cwe/cwe-532
 *       external/cwe/cwe-550
 * @cwe CWE-550
 */

import java
import semmle.code.java.dataflow.TaintTracking
import CommonSinks.CommonSinks

module Flow = TaintTracking::Global<SqlExceptionToConsoleConfig>;
import Flow::PathGraph

 // Configuration for tracking the flow of sensitive information from SQLExceptions to the console.
module SqlExceptionToConsoleConfig implements DataFlow::ConfigSig {
 

  // Sources are calls to methods on SQLException that retrieve error messages or codes.
  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      (mc.getReceiverType().(RefType).hasQualifiedName("java.sql", "SQLException") or
      mc.getReceiverType().(RefType).getASupertype*().hasQualifiedName("java.sql", "SQLException")or 
      // Extend sources to include common SQL-related exceptions from popular libraries
      mc.getReceiverType().(RefType).hasQualifiedName("org.springframework.jdbc", "DataAccessException") or
      mc.getReceiverType().(RefType).hasQualifiedName("org.hibernate", "HibernateException")) and
      mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "getSQLState", 
      "getErrorCode", "toString", "getenv", "getProperty", "getParameter", "getAttribute"]) and
      source.asExpr() = mc
    )
  }
  
  predicate isSink(DataFlow::Node sink) {
    CommonSinks::isPrintSink(sink) or
    CommonSinks::isLoggingSink(sink) or
    CommonSinks::isServletSink(sink) or
    CommonSinks::isErrorSink(sink) or
    CommonSinks::isIOSink(sink)
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-550: (SQL) Server Generated Error Message Containing Sensitive Information."
