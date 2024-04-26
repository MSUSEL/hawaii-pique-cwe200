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
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;
import Flow::PathGraph

module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {

  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      // Sources from exceptions
      mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      (mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace", "toString"])) and
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
    exists(CatchClause cc, MethodCall mc |
      // Ensure the CatchClause is catching ServletException
      cc.getACaughtType().hasQualifiedName("javax.servlet", "ServletException") and
      // Ensure the MethodCall is within the CatchClause for the ServletException
      mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
      // Ensure the sink matches one of the known sensitive sinks
      (
        CommonSinks::isLoggingSink(sink) or
        CommonSinks::isPrintSink(sink) or
        CommonSinks::isServletSink(sink) or
        CommonSinks::isErrorSink(sink) or
        CommonSinks::isIOSink(sink)
      ) and
      // Link the sink to the argument of the MethodCall
      sink.asExpr() = mc.getAnArgument()
    )

    or
    exists(ConstructorCall cc |
      cc.getConstructedType().hasQualifiedName("javax.servlet", "ServletException") and
      sink.asExpr() = cc.getAnArgument()
    )
}

}



from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
