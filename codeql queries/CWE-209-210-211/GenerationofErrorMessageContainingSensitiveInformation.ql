/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/error-message-sensitive-info/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 import DataFlow::PathGraph
 import SensitiveInfo.SensitiveInfo
 import CommonSinks.CommonSinks
 
 // Define flow states
 class State1 extends DataFlow::FlowState { State1() { this = "State1" } }
 class State2 extends DataFlow::FlowState { State2() { this = "State2" } }
 class State3 extends DataFlow::FlowState { State3() { this = "State3" } }
 
 // Dataflow configuration using a manual link for throw/catch
 class SensitiveInfoInErrorMsgConfig extends TaintTracking::Configuration {
   SensitiveInfoInErrorMsgConfig() { this = "SensitiveInfoInErrorMsgConfig" }
 
   // Track sensitive variables as the source in State1
   override predicate isSource(DataFlow::Node source, DataFlow::FlowState state) {
     state instanceof State1 and
     exists(SensitiveVariableExpr sve |
       source.asExpr() = sve and
       (
        sve.toString() != "e" and
        sve.toString() != "ex"
       )
     )
   }
 
   // Track sinks like `println`, `sendError`, etc. in State3
   override predicate isSink(DataFlow::Node sink, DataFlow::FlowState state) {
     state instanceof State3 and
     exists(MethodCall mcSink |
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isServletSink(sink) or
         CommonSinks::isLoggingSink(sink)
       ) and 
       sink.asExpr() = mcSink.getAnArgument()
     )
   }
 
   // Define transitions between flow states
   override predicate isAdditionalTaintStep(
     DataFlow::Node node1, DataFlow::FlowState state1,
     DataFlow::Node node2, DataFlow::FlowState state2
   ) {
     // Transition from State1 to State2: sensitive data flows into any exception constructor, except RuntimeException and ServletException (and their subclasses)
     state1 instanceof State1 and
     state2 instanceof State2 and
     exists(ConstructorCall cc |
       cc.getAnArgument() = node1.asExpr() and
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "Throwable") and
       // Exclude RuntimeException and its subclasses
       not cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
       // Exclude ServletException and its subclasses
       not cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("javax.servlet", "ServletException") and
       cc = node2.asExpr()
     ) or
 
     // Transition from State2 to State3: manually link throw to catch when in the same method
     state1 instanceof State2 and
     state2 instanceof State3 and
     exists(ThrowStmt t, CatchClause catchClause, MethodCall mcGetMessage |
       t.getExpr() = node1.asExpr() and
       catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
       catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
       mcGetMessage.getMethod().getName() = "getMessage" and
       node2.asExpr() = mcGetMessage
     )
   }
 }
 
 // Query for sensitive information flow from source to sink with path visualization
 from DataFlow::PathNode source, DataFlow::PathNode sink, SensitiveInfoInErrorMsgConfig cfg
 where cfg.hasFlowPath(source, sink)
 select sink, source, sink, "Sensitive information flows into exception and is exposed via getMessage."
 