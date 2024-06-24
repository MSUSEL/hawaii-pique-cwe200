/**
 * @name CWE-550: Exposure of sensitive SQL exception information
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
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<SqlExceptionToConsoleConfig>;
import Flow::PathGraph

 // Configuration for tracking the flow of sensitive information from SQLExceptions to the console.
module SqlExceptionToConsoleConfig implements DataFlow::ConfigSig {
 

  // Sources are calls to methods on SQLException that retrieve error messages or codes.
  predicate isSource(DataFlow::Node source) {
    exists(CatchClause cc | 
      // Check if the exception is a SQLException or a subclass
      (
      cc.getVariable().getType().(RefType).getASupertype*().hasQualifiedName("java.sql", "SQLException") or
      cc.getVariable().getType().(RefType).hasQualifiedName("org.springframework.jdbc", "DataAccessException") or
      cc.getVariable().getType().(RefType).hasQualifiedName("org.hibernate", "HibernateException") 
      )
      and
      (
        // Direct access to the exception variable
        source.asExpr() = cc.getVariable().getAnAccess() or
        // Access to methods called on the exception object
        exists(MethodCall mc | 
          mc.getQualifier() = cc.getVariable().getAnAccess() and
          source.asExpr() = mc
        )
      )
    )
  }
  
  predicate isSink(DataFlow::Node sink) {
    CommonSinks::isPrintSink(sink) or
    CommonSinks::isServletSink(sink) or
    CommonSinks::isErrorSink(sink) or
    CommonSinks::isIOSink(sink) or
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
select sink.getNode(), source, sink, "CWE-550: (SQL) Server Generated Error Message Containing Sensitive Information."
