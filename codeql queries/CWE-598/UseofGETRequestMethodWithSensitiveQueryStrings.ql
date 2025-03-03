/**
 * @name CWE-598: Use of GET Request Method with Sensitive Query Strings
 * @description Detects sensitive information being sent in query strings over GET requests, which could be exposed in server logs or browser history.
 * @kind path-problem
 * @problem.severity warning
 * @id java/sensitive-get-query/598
 * @tags security
 *       external/cwe/cwe-598
 * @cwe CWE-598
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import SensitiveInfo.SensitiveInfo
import Barrier.Barrier

module Flow = TaintTracking::Global<SensitiveInfoToUrlConfig>;
import Flow::PathGraph

module SensitiveInfoToUrlConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
  }

  predicate isSink(DataFlow::Node sink) {

    // Case 1: Explicit GET via HttpGet constructor (Apache HttpClient)
    exists(ConstructorCall httpGetCall |
      httpGetCall.getConstructedType().hasQualifiedName("org.apache.http.client.methods", "HttpGet") and
      DataFlow::localExprFlow(sink.asExpr(), httpGetCall.getAnArgument())
    )
    or
    // Case 2: Explicit GET via URL constructor + openConnection + explicit setRequestMethod("GET")
    exists(ConstructorCall urlConstructor, MethodCall openConnectionCall, MethodCall setRequestMethod |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(sink.asExpr(), urlConstructor.getAnArgument()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
    or
    // Case 3: Explicit GET via URI-to-URL conversion + openConnection + explicit setRequestMethod("GET")
    exists(ConstructorCall uriConstructor, MethodCall toUrlCall, MethodCall openConnectionCall, MethodCall setRequestMethod |
      uriConstructor.getConstructedType().hasQualifiedName("java.net", "URI") and
      DataFlow::localExprFlow(sink.asExpr(), uriConstructor.getAnArgument()) and
      toUrlCall.getMethod().hasName("toURL") and
      DataFlow::localExprFlow(uriConstructor, toUrlCall.getQualifier()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "HttpURLConnection") and
      DataFlow::localExprFlow(toUrlCall, openConnectionCall.getQualifier()) and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
    or
    // Case 4: Default GET via URL constructor + openConnection (no explicit setRequestMethod)
    exists(ConstructorCall urlConstructor, MethodCall openConnectionCall |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(sink.asExpr(), urlConstructor.getAnArgument()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
      not exists(MethodCall setRequestMethod |
        setRequestMethod.getMethod().hasName("setRequestMethod") and
        DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
      )
    )
    or
    // Case 5: Default GET via URI-to-URL conversion + openConnection (no explicit setRequestMethod)
    exists(ConstructorCall uriConstructor, MethodCall toUrlCall, MethodCall openConnectionCall |
      uriConstructor.getConstructedType().hasQualifiedName("java.net", "URI") and
      DataFlow::localExprFlow(sink.asExpr(), uriConstructor.getAnArgument()) and
      toUrlCall.getMethod().hasName("toURL") and
      DataFlow::localExprFlow(uriConstructor, toUrlCall.getQualifier()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "HttpURLConnection") and
      DataFlow::localExprFlow(toUrlCall, openConnectionCall.getQualifier()) and
      not exists(MethodCall setRequestMethod |
        setRequestMethod.getMethod().hasName("setRequestMethod") and
        DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
      )
    )
  }

  predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
  }
}

/**
 * Checks if an expression contains a string literal with a query parameter indicator,
 * e.g., "?param=" or "&param=".
 */
predicate containsQueryParamIndicator(Expr e) {
  exists(StringLiteral s |
    s = e and
    s.getValue().regexpMatch(".*[?&][^=]*=.*")
  )
  or
  exists(AddExpr add |
    add = e and
    (containsQueryParamIndicator(add.getLeftOperand()) or containsQueryParamIndicator(add.getRightOperand()))
  )
}

from Flow::PathNode source, Flow::PathNode sink
where
  Flow::flowPath(source, sink) and
  containsQueryParamIndicator(sink.getNode().asExpr())
select sink.getNode(), source, sink,
  "Sensitive information used in a URL query string parameter for a GET request."
