/**
 * @name Use of GET request method with sensitive query strings
 * @description Detects sensitive information being sent in query strings over GET requests, which could be exposed in server logs or browser history.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-598
 * @tags security
 *       external/cwe/cwe-598
 * @cwe CWE-598
 */

import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph

/** A configuration for finding flows from sensitive information sources to URL constructions. */
class SensitiveInfoToUrlConfig extends TaintTracking::Configuration {
  SensitiveInfoToUrlConfig() { this = "SensitiveInfoToUrlConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |  source.asExpr() = sve)
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(ConstructorCall urlConstructor |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      urlConstructor.getAnArgument() = sink.asExpr() and
      // Find the usage of the URL in an openConnection call
      exists(MethodAccess openConnection |
        openConnection.getMethod().hasName("openConnection") and
        DataFlow::localExprFlow(urlConstructor, openConnection.getQualifier()) and
        // Ensure this connection is used in a setRequestMethod call with "GET"
        exists(MethodAccess setRequestMethod |
          setRequestMethod.getMethod().hasName("setRequestMethod") and
          setRequestMethod.getArgument(0).(StringLiteral).getValue() = "GET" and
          DataFlow::localExprFlow(openConnection, setRequestMethod.getQualifier())
        )
      )
    )
  }
}

from SensitiveInfoToUrlConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive information used in a URL constructed for a GET request."
