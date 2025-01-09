/**
 * @name CWE-598: Sensitive Information in Query String in GET Request
 * @description Detects sensitive information being used as query parameters in URLs for HTTP GET requests.
 * @kind path-problem
 * @problem.severity error
 * @id java/http-query-sensitive-info/598
 * @tags security
 *       external/cwe/cwe-598
 * @cwe CWE-598
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 import DataFlow::PathGraph
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 // Define flow states
 class SensitiveState extends DataFlow::FlowState { SensitiveState() { this = "SensitiveState" } }
 class QueryParamState extends DataFlow::FlowState { QueryParamState() { this = "QueryParamState" } }
 class GetRequestState extends DataFlow::FlowState { GetRequestState() { this = "GetRequestState" } }
 
 // Dataflow configuration
 class SensitiveToUrlConfig extends TaintTracking::Configuration {
   SensitiveToUrlConfig() { this = "SensitiveToUrlConfig" }
 
   // Identify sensitive sources in SensitiveState
   override predicate isSource(DataFlow::Node source, DataFlow::FlowState state) {
     state instanceof SensitiveState and
     exists(SensitiveVariableExpr sve |
       source.asExpr() = sve and
       not sve.toString().toLowerCase().matches("%url%")
     )
   }
 
   // Identify GET request sinks in GetRequestState
   override predicate isSink(DataFlow::Node sink, DataFlow::FlowState state) {
     state instanceof GetRequestState and
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
   }
 
   // Define transitions between flow states
   override predicate isAdditionalTaintStep(
     DataFlow::Node node1, DataFlow::FlowState state1,
     DataFlow::Node node2, DataFlow::FlowState state2
   ) {
     // Transition from SensitiveState to QueryParamState
     state1 instanceof SensitiveState and
     state2 instanceof QueryParamState and
     exists(AddExpr concatOp, Expr param |
       (param = concatOp.getLeftOperand() or param = concatOp.getRightOperand()) and
       (param.toString().indexOf("?") >= 0 or param.toString().indexOf("&") >= 0) and
       param.toString().indexOf("=") > 0 and
       node1.asExpr() = param and
       node2.asExpr() = concatOp
     ) or
 
     // Transition from QueryParamState to GetRequestState
     state1 instanceof QueryParamState and
     state2 instanceof GetRequestState and
     exists(ConstructorCall urlConstructor |
       urlConstructor.getAnArgument() = node1.asExpr() and
       node2.asExpr() = urlConstructor
     )
   }
 
   // Barriers block the flow
   override predicate isSanitizer(DataFlow::Node node) {
     Barrier::barrier(node)
   }
 }
 
 // Main query for detecting the flow
 from DataFlow::PathNode source, DataFlow::PathNode sink, SensitiveToUrlConfig cfg
 where cfg.hasFlowPath(source, sink)
 select sink, source, sink, "Sensitive information flows through query parameters into a GET request."
 