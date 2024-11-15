/**
 * @name CWE-598: Use of GET request method with sensitive query strings
 * @description Detects sensitive information being sent in query strings over GET requests, which could be exposed in server logs or browser history.
 * @kind path-problem
 * @problem.severity warning
 * @id java/sensitive-get-query2/598
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
 
 /** A configuration for finding flows from sensitive information sources to URL constructions. */
 module SensitiveInfoToUrlConfig implements DataFlow::ConfigSig {
 
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve |  
      source.asExpr() = sve and 
      not sve.toString().toLowerCase().matches("%url%"))
    }
 
   predicate isSink(DataFlow::Node sink) {
    // Direct use of URL with openConnection followed by setRequestMethod("GET")
    exists(ConstructorCall urlConstructor, MethodCall openConnectionCall, MethodCall setRequestMethod |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      urlConstructor.getAnArgument() = sink.asExpr() and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
    or
    // Handling URI -> URL conversion and subsequent GET request
    exists(ConstructorCall uriConstructor, MethodCall toUrlCall, MethodCall openConnectionCall, MethodCall setRequestMethod |
      uriConstructor.getConstructedType().hasQualifiedName("java.net", "URI") and
      uriConstructor.getAnArgument() = sink.asExpr() and
      toUrlCall.getMethod().hasName("toURL") and
      toUrlCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URI") and
      DataFlow::localExprFlow(uriConstructor, toUrlCall.getQualifier()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
      DataFlow::localExprFlow(toUrlCall, openConnectionCall.getQualifier()) and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
    or
    // Apache HttpClient usage with HttpGet and execute method call
    exists(ConstructorCall httpGetCall |
      httpGetCall.getConstructedType().hasQualifiedName("org.apache.http.client.methods", "HttpGet") and
      httpGetCall.getAnArgument() = sink.asExpr()
    )
    or
    // HttpURLConnection directly initialized with GET method
    exists(MethodCall openConnectionCall, MethodCall setRequestMethod |
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "HttpURLConnection") and
      DataFlow::localExprFlow(sink.asExpr(), openConnectionCall.getQualifier()) and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
  }
 
   predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "Sensitive information used in a URL constructed for a GET request." 
 
 