/**
 * @name CWE-536: Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-runtime-error-message-exposure-direct/536
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
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;
 
 import Flow::PathGraph
 
 module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   predicate isSink(DataFlow::Node sink) {
     exists(CatchClause cc, MethodCall mc |
       // Ensure the CatchClause is catching ServletException
       cc.getACaughtType().hasQualifiedName("javax.servlet", "ServletException") and
       // Ensure the MethodCall is within the CatchClause for the ServletException
       mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
       // Ensure the sink matches one of the known sensitive sinks
       (
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isServletSink(sink) or
         CommonSinks::isErrorSink(sink) or
         CommonSinks::isIOSink(sink) or
         getSinkAny(sink)
       ) and
       // Link the sink to the argument of the MethodCall
       sink.asExpr() = mc.getAnArgument()
     )
   }
 
  predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink,
   "CWE-536: Servlet Runtime Error Message Containing Sensitive Information."
 