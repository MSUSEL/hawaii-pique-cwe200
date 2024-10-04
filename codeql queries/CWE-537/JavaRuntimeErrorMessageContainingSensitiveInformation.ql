/**
 * @name CWE-537: Java Runtime Error Message Containing Sensitive Information
 * @description Detects sensitive information (e.g., apiKey) being added to an exception message and then exposed via getMessage in HTTP responses.
 * @kind path-problem
 * @problem.severity error
 * @id java/runtime-error-message-exposure/537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 import DataFlow::PathGraph
 
 // Define sensitive variables
 class SensitiveVariable extends VarAccess {
   SensitiveVariable() {
     this.getVariable().getName() = "username" or
     this.getVariable().getName() = "email" or
     this.getVariable().getName() = "password" or
     this.getVariable().getName() = "apiKey"
   }
 }
 
 // Define flow states
 class State1 extends DataFlow::FlowState { State1() { this = "State1" } }
 class State2 extends DataFlow::FlowState { State2() { this = "State2" } }
 class State3 extends DataFlow::FlowState { State3() { this = "State3" } }
 
 // Dataflow configuration using a manual link for throw/catch
 class ExceptionDataFlowConfig extends TaintTracking::Configuration {
   ExceptionDataFlowConfig() { this = "ExceptionDataFlowConfig" }
 
   // Track sensitive variables as the source in State1
   override predicate isSource(DataFlow::Node source, DataFlow::FlowState state) {
     state instanceof State1 and
     exists(SensitiveVariable var |
       source.asExpr() = var
     )
   }
 
   // Track sinks like `println`, `sendError`, etc. in State3
   override predicate isSink(DataFlow::Node sink, DataFlow::FlowState state) {
     state instanceof State3 and
     exists(MethodCall mcSink |
       mcSink.getMethod().getName() in ["println", "sendError", "write", "sendError"] and
       mcSink.getAnArgument() = sink.asExpr()
     )
   }
 
   // Define transitions between flow states
   override predicate isAdditionalTaintStep(
     DataFlow::Node node1, DataFlow::FlowState state1,
     DataFlow::Node node2, DataFlow::FlowState state2
   ) {
     // Transition from State1 to State2: sensitive data flows into an exception constructor
     state1 instanceof State1 and
     state2 instanceof State2 and
     exists(ConstructorCall cc |
       cc.getAnArgument() = node1.asExpr() and
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
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
 from DataFlow::PathNode source, DataFlow::PathNode sink, ExceptionDataFlowConfig cfg
 where cfg.hasFlowPath(source, sink)
 select sink, source, sink, "Sensitive information flows into exception and is exposed via getMessage."
 