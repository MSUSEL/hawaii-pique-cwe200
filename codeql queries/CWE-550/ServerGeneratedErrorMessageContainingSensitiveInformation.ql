/**
 * @name CWE-550: Exposure of sensitive information through servlet responses
 * @description Detects when sensitive information from exceptions or system details
 *              is exposed to clients via servlet responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 import SensitiveInfo.SensitiveInfo
 import CommonSinks.CommonSinks
 import Barrier.Barrier
 
 // Define flow states
 private newtype MyFlowState =
 State1() or
 State2() or
 State3()
 
 // Dataflow configuration using a manual link for throw/catch
 module ServerGeneratedErrorMessageConfig implements DataFlow::StateConfigSig {
 
  class FlowState = MyFlowState;

   // Track sensitive variables as the source in State1
  predicate isSource(DataFlow::Node source, FlowState state) {
     state instanceof State1 and
     exists(SensitiveVariableExpr sve |
       source.asExpr() = sve
     )
   }
 
   // Track sinks like `println`, `sendError`, etc. in State3
  predicate isSink(DataFlow::Node sink, FlowState state) {
     state instanceof State3 and
     exists(MethodCall mcSink |
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isServletSink(sink) 
        //  or
        //  getSinkAny(sink) 
       ) and 
         sink.asExpr() = mcSink.getAnArgument() 
      )
    }
 
   // Define transitions between flow states
   predicate isAdditionalFlowStep(
     DataFlow::Node node1, FlowState state1,
     DataFlow::Node node2, FlowState state2
   ) {
     // Transition from State1 to State2: sensitive data flows into a runtime exception constructor
     state1 instanceof State1 and
     state2 instanceof State2 and
     exists(ConstructorCall cc |
       cc.getAnArgument() = node1.asExpr() and
       (
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.sql", "SQLException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("org.springframework.web.client", "HttpServerErrorException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("org.springframework.web.server", "ResponseStatusException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("org.apache.http", "ApplicationServerException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.net", "BindException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.net", "ConnectException") or 
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.net", "SocketTimeoutException")  or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.net", "ProtocolException") or 
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("org.springframework.boot.web.server", "WebServerException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("org.springframework.http.converter", "HttpMessageNotWritableException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("javax.jms", "JMSException") or
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("javax.ejb", "EJBException")
       )
       and
       cc = node2.asExpr()
     ) or
 
     // Transition from State2 to State3: link throw to catch for getMessage()
     state1 instanceof State2 and
     state2 instanceof State3 and
     exists(ThrowStmt t, CatchClause catchClause, MethodCall mcGetMessage |
       t.getExpr() = node1.asExpr() and
       catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
       catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
       mcGetMessage.getMethod().getName() = "getMessage" and
       node2.asExpr() = mcGetMessage
     ) or
 
     // Handle cases where the exception object itself (e) is passed to a method like print(e)
     state1 instanceof State2 and
     state2 instanceof State3 and
     exists(ThrowStmt t, CatchClause catchClause |
       t.getExpr() = node1.asExpr() and
       catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
       node2.asExpr() = catchClause.getVariable().getAnAccess()
     )
   }

   predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
   }
 }
 

 module SensitiveInfoInErrorMsgFlow = TaintTracking::GlobalWithState<ServerGeneratedErrorMessageConfig>;
 import SensitiveInfoInErrorMsgFlow::PathGraph
 

 // Query for sensitive information flow from source to sink with path visualization
 from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
 where SensitiveInfoInErrorMsgFlow::flowPath(source, sink)
 select sink, source, sink,
   "CWE-550: Sensitive information is exposed via a server error message."
 