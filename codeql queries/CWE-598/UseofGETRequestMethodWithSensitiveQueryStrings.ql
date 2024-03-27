import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph

/**
 * @name Use of GET request method with sensitive query strings
 * @description Detects sensitive information being sent in query strings over GET requests, which could be exposed in server logs or browser history.
 * @kind path-problem
 * @problem.severity warning
 * @id java/cwe-598-general
 * @tags security
 *       external/cwe/cwe-598
 */

/** A configuration for finding flows of sensitive information into URLs used in GET requests. */
class SensitiveInfoInGetRequestConfig extends TaintTracking::Configuration {
  SensitiveInfoInGetRequestConfig() { this = "SensitiveInfoInGetRequestConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    // Focus on where URL objects are used to open a connection, ensuring the connection is configured for a GET request.
    exists(MethodAccess ma, VarAccess va |
      ma.getMethod().hasName("openConnection") and
      va.getVariable().getType().(RefType).hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(va, ma.getQualifier()) and
      exists(MethodAccess setMethod |
        setMethod.getMethod().hasName("setRequestMethod") and
        setMethod.getMethod().getDeclaringType().hasQualifiedName("java.net", "HttpURLConnection") and
        setMethod.getArgument(0).(StringLiteral).getValue() = "GET" and
        DataFlow::localExprFlow(ma, setMethod.getQualifier())
      )
      |
      sink.asExpr() = va
    )
  }
}

from SensitiveInfoInGetRequestConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "CWE-598: Sensitive information exposed via GET request query string."
