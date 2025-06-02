/**
 * @name CWE-537: Java Runtime Error Message Containing Sensitive Information
 * @description Detects sensitive information (e.g., apiKey) being added to an exception message and then exposed via getMessage or the exception object in HTTP responses.
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
 import SensitiveInfo.SensitiveInfo
 import CommonSinks.CommonSinks
 import Barrier.Barrier

 private newtype MyFlowState =
 State1() or
 State2() or
 State3()
 
 
 // Dataflow configuration using a manual link for throw/catch
 module ExceptionDataFlowConfig implements DataFlow::StateConfigSig { 
   
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
       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
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
 

 predicate isTestFile(File f) {
  // Convert path to lowercase for case-insensitive matching
  exists(string path | path = f.getAbsolutePath().toLowerCase() |
    // Check for common test-related directory or file name patterns
    path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
  )
}

// Instantiate a GlobalWithState for taint tracking
module SensitiveInfoInErrorMsgFlow = 
TaintTracking::GlobalWithState<ExceptionDataFlowConfig>;

// Import its PathGraph (not the old DataFlow::PathGraph)
import SensitiveInfoInErrorMsgFlow::PathGraph

// Use the flowPath from that module
from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
where SensitiveInfoInErrorMsgFlow::flowPath(source, sink) and
not isTestFile(sink.getNode().getLocation().getFile())
select sink, source, sink,
"CWE-537: Sensitive information flows into exception and is exposed via an error message."

 