/**
 * @name Use of GET request method with sensitive query strings
 * @description Detects sensitive information being sent in query strings over GET requests, which could be exposed in server logs or browser history.
 * @kind path-problem
 * @problem.severity warning
 * @id ava/sensitive-get-query2/598
 * @tags security
 *       external/cwe/cwe-598
 * @cwe CWE-598
 */

import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables

module Flow = TaintTracking::Global<SensitiveInfoToUrlConfig>;
import Flow::PathGraph

/** A configuration for finding flows from sensitive information sources to URL constructions. */
module SensitiveInfoToUrlConfig implements DataFlow::ConfigSig {

  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |  source.asExpr() = sve)
  }

  predicate isSink(DataFlow::Node sink) {
    exists(ConstructorCall urlConstructor |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      urlConstructor.getAnArgument() = sink.asExpr() and
      // Find the usage of the URL in an openConnection call
      exists(MethodCall openConnection |
        openConnection.getMethod().hasName("openConnection") and
        DataFlow::localExprFlow(urlConstructor, openConnection.getQualifier()) and
        // Ensure this connection is used in a setRequestMethod call with "GET"
        exists(MethodCall setRequestMethod |
          setRequestMethod.getMethod().hasName("setRequestMethod") and
          setRequestMethod.getArgument(0).(StringLiteral).getValue() = "GET" and
          DataFlow::localExprFlow(openConnection, setRequestMethod.getQualifier())
        )
      )
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive information used in a URL constructed for a GET request." 

