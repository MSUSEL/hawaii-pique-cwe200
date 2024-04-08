/**
 * @name Generation of Error Message Containing Sensitive Information (CWE-209)
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/error-message-sensitive-info/CWE-209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.security.SensitiveVariables

module Flow = TaintTracking::Global<SensitiveInfoInErrorMsgConfig>;
import Flow::PathGraph
/** A configuration for tracking sensitive information flow into error messages. */
module SensitiveInfoInErrorMsgConfig implements DataFlow::ConfigSig{
  predicate isSource(DataFlow::Node source) {
    // Broad definition, consider refining
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)   
  }

  predicate isSink(DataFlow::Node sink) {
    // Identifying common error message generation points
    exists(MethodCall mc | 
      mc.getMethod().getName().regexpMatch("printStackTrace|log|error|println") and
      sink.asExpr() = mc.getArgument(0)
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-209: Error message may contain sensitive information."
