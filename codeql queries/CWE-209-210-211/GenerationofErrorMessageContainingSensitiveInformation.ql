/**
 * @name Generation of Error Message Containing Sensitive Information (CWE-209)
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph

/** A configuration for tracking sensitive information flow into error messages. */
class SensitiveInfoInErrorMsgConfig extends TaintTracking::Configuration {
  SensitiveInfoInErrorMsgConfig() { this = "SensitiveInfoInErrorMsgConfig" }

  override predicate isSource(DataFlow::Node source) {
    // Broad definition, consider refining
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)   
  }

  override predicate isSink(DataFlow::Node sink) {
    // Identifying common error message generation points
    exists(MethodAccess ma | 
      ma.getMethod().getName().regexpMatch("printStackTrace|log|error|println") and
      sink.asExpr() = ma.getArgument(0)
    )
  }
}

from SensitiveInfoInErrorMsgConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "CWE-209: Error message may contain sensitive information."
