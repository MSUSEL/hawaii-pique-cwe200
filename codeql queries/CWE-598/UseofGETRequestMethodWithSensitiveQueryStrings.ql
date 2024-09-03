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
     or
     // Detect use of Apache HttpGet that flows into an execute method call
     exists(ConstructorCall httpGetCall |
       httpGetCall.getConstructedType().hasQualifiedName("org.apache.http.client.methods", "HttpGet") and
       httpGetCall.getAnArgument() = sink.asExpr() 
     ) 
     or
     // Detect use of java.net.URL that flows into an openConnection method call
     exists(ConstructorCall urlConstructor, MethodCall openConnectionCall |
       urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
       urlConstructor.getAnArgument() = sink.asExpr() and
       // Ensure the URL instance flows to an openConnection method
       DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
       // The default request method is GET
       openConnectionCall.getMethod().hasName("openConnection") and
       openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL")
     ) 
     or
     // Handling cases where URI is converted to URL and used in HttpURLConnection
     exists(ConstructorCall uriConstructor, MethodCall toUrlCall, MethodCall openConnectionCall, MethodCall setRequestMethod |
       uriConstructor.getConstructedType().hasQualifiedName("java.net", "URI") and
       uriConstructor.getAnArgument() = sink.asExpr() and
       toUrlCall.getMethod().hasName("toURL") and
       toUrlCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URI") and
       DataFlow::localExprFlow(uriConstructor, toUrlCall.getQualifier()) and
       DataFlow::localExprFlow(toUrlCall, openConnectionCall.getQualifier()) and
       openConnectionCall.getMethod().hasName("openConnection") and
       openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
       setRequestMethod.getMethod().hasName("setRequestMethod") and
       ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
       DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
     ) 
     or
     exists(ConstructorCall urlConstructor, MethodCall openConnectionCall, MethodCall setRequestMethod |
      urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
      urlConstructor.getAnArgument() = sink.asExpr() and
      DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
      openConnectionCall.getMethod().hasName("openConnection") and
      openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
      setRequestMethod.getMethod().hasName("setRequestMethod") and
      setRequestMethod.getArgument(0) instanceof StringLiteral and
      ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
      DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
    )
   }
 
   predicate isBarrier(DataFlow::Node node) {
     exists(MethodCall mc |
       // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
       (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
       mc.getMethod().getName().toLowerCase().matches("%encrypt%")) and
     // Consider both arguments and the return of sanitization/encryption methods as barriers
     (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
     )
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "Sensitive information used in a URL constructed for a GET request." 
 
 