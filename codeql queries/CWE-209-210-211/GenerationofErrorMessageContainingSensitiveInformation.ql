/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/error-message-sensitive-info/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

 import java
 private import semmle.code.java.dataflow.ExternalFlow
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<SensitiveInfoInErrorMsgConfig>;
 import Flow::PathGraph
 /** A configuration for tracking sensitive information flow into error messages. */
 module SensitiveInfoInErrorMsgConfig implements DataFlow::ConfigSig{
   predicate isSource(DataFlow::Node source) {
     // Broad definition, consider refining
     exists(SensitiveVariableExpr sve | source.asExpr() = sve) or 
     exists(SensitiveStringLiteral ssl |source.asExpr() = ssl) 
    //  or
     
    //  exists(MethodCall ma |
    //    ma.getMethod().hasName("getMessage") and
    //    source.asExpr() = ma
    //  )
   }
 
   predicate isSink(DataFlow::Node sink) {
     // Identifying common error message generation points
     CommonSinks::isPrintSink(sink) or 
     CommonSinks::isErrorSink(sink)
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "CWE-209: Error message may contain sensitive information."
 