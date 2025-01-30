/**
 * @name CWE-598: Use of GET Request Method with Sensitive Query Strings
 * @description Detects sensitive data in GET request query parameters
 * @kind path-problem
 * @id java/sensitive-get-query/598
 * @tags security
 */

 import java
 import semmle.code.java.dataflow.DataFlow
 import semmle.code.java.dataflow.TaintTracking
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<SensitiveInfoToUrlConfig>;
 import Flow::PathGraph
 
 module SensitiveInfoToUrlConfig implements DataFlow::ConfigSig {
 
   // Define sources: Sensitive data variables that might flow into URLs
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve |
       source.asExpr() = sve
     )
   }
 
   // Define sinks: URLs containing query parameters in an HttpGet request
   predicate isSink(DataFlow::Node sink) {
     exists(ConstructorCall httpGetCall, AddExpr ae, Literal queryLiteral |
    //    httpGetCall.getConstructedType().hasQualifiedName("org.apache.http.client.methods", "HttpGet") and
    //    httpGetCall.getAnArgument() = ae and
       queryLiteral = ae.getAnOperand() and
       queryLiteral.getValue().regexpMatch(".*\\?.*=.*") // Ensures it's a query parameter
       and sink.asExpr() = ae
     )
   }
 
   predicate isBarrier(DataFlow::Node node) {
     Barrier::barrier(node)
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, 
   "Sensitive data in GET query parameter of an HttpGet request: " + source.getNode().toString()
 