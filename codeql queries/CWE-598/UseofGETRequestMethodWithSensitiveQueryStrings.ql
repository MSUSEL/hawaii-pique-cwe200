import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph

/**
 * @name Use of GET request method with potential sensitive query strings
 * @description Using GET request with sensitive data in the query string can expose the information.
 * @kind path-problem
 * @problem.severity warning
 * @id java/cwe-598
 * @tags security
 *       external/cwe/cwe-598
 */
class GetWithSensitiveDataQuery extends DataFlow::Configuration {
  GetWithSensitiveDataQuery() { this = "GetWithSensitiveDataQuery" }
  
  override predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
        source.asExpr() = sve
      )
  }
  
  override predicate isSink(DataFlow::Node sink) {
    exists(MethodCall ma |
      ma.getMethod().getDeclaringType().hasQualifiedName("java.net", "HttpURLConnection") and
    //   ma.getMethod().getName() = "setRequestMethod" and
      ma.getArgument(0).(StringLiteral).getValue() = "GET" and
      sink.asExpr() = ma.getQualifier()
    )
  }
}

from GetWithSensitiveDataQuery config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive data might be exposed via GET request."
