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
 
   // Track sensitive variables or request parameters as sources in State1
   predicate isSource(DataFlow::Node source, FlowState state) {
     state instanceof State1 and
     (
       exists(SensitiveVariableExpr sve | source.asExpr() = sve) or
       // Include request.getParameter results as sensitive
       exists(MethodCall mc |
         mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletRequest", "getParameter") and
         source.asExpr() = mc
       )
     )
   }
 
   // Track sinks like println, write, sendError, etc. in State3
   predicate isSink(DataFlow::Node sink, FlowState state) {
     state instanceof State3 and
     exists(MethodCall mcSink |
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isServletSink(sink) or
         // Explicitly include PrintWriter.println and write
         mcSink.getMethod().hasQualifiedName("java.io", "PrintWriter", ["println", "write"]) and
         mcSink.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter")
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
     (
       state1 instanceof State1 and
       state2 instanceof State2 and
       exists(ConstructorCall cc |
         cc.getAnArgument() = node1.asExpr() and
         cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
         cc = node2.asExpr()
       )
     ) or
     // Transition from State2 to State3: link throw to catch for getMessage()
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       (
         // Case 1: Same-method throw to catch via getMessage()
         exists(ThrowStmt t, CatchClause catchClause, MethodCall mcGetMessage |
           t.getExpr() = node1.asExpr() and
           catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
           catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
           mcGetMessage.getMethod().getName() = "getMessage" and
           node2.asExpr() = mcGetMessage
         ) or
         // Case 2: Cross-method throw to catch via getMessage() (new)
         exists(
           ThrowStmt t,
           CatchClause catchClause,
           MethodCall mcGetMessage,
           Call call |
           t.getExpr() = node1.asExpr() and
           call.getCallee() = t.getEnclosingCallable() and
           catchClause.getEnclosingCallable() = call.getEnclosingCallable() and
           catchClause.getVariable().getType().(RefType).hasSubtype*(t.getExpr().getType()) and
           catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
           mcGetMessage.getMethod().getName() = "getMessage" and
           node2.asExpr() = mcGetMessage
         )
       )
     ) or
     // Handle cases where the exception object itself (e) is passed to a method
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       (
         // Case 3: Same-method throw to catch via exception object
         exists(ThrowStmt t, CatchClause catchClause |
           t.getExpr() = node1.asExpr() and
           catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
           node2.asExpr() = catchClause.getVariable().getAnAccess()
         ) or
         // Case 4: Cross-method throw to catch via exception object (new)
         exists(
           ThrowStmt t,
           CatchClause catchClause,
           Call call |
           t.getExpr() = node1.asExpr() and
           call.getCallee() = t.getEnclosingCallable() and
           catchClause.getEnclosingCallable() = call.getEnclosingCallable() and
           catchClause.getVariable().getType().(RefType).hasSubtype*(t.getExpr().getType()) and
           node2.asExpr() = catchClause.getVariable().getAnAccess()
         )
       )
     ) or
     // State3 to State3: Parameter flow to expression use (new)
     (
       state1 instanceof State3 and
       state2 instanceof State3 and
       exists(MethodCall mc, DataFlow::Node paramUse, Parameter param |
         node1.asExpr() = mc.getAnArgument() and
         mc.getCallee().getAParameter() = param and
         node2 = DataFlow::parameterNode(param) and
         DataFlow::localFlowStep(node2, paramUse) and
         paramUse.asExpr() instanceof Expr
       )
     ) or
     // State3 to State3: String concatenation (new)
     (
       state1 instanceof State3 and
       state2 instanceof State3 and
       exists(AddExpr add |
         node2.asExpr() = add and
         (node1.asExpr() = add.getLeftOperand() or node1.asExpr() = add.getRightOperand())
       )
     )
   }
 
   predicate isBarrier(DataFlow::Node node) {
     Barrier::barrier(node)
   }
 }
 
 predicate isTestFile(File f) {
   exists(string path | path = f.getAbsolutePath().toLowerCase() |
     path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
   )
 }
 
 // Instantiate a GlobalWithState for taint tracking
 module SensitiveInfoInErrorMsgFlow = TaintTracking::GlobalWithState<ExceptionDataFlowConfig>;
 
 // Import its PathGraph
 import SensitiveInfoInErrorMsgFlow::PathGraph
 
 // Use the flowPath from that module
 from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
 where SensitiveInfoInErrorMsgFlow::flowPath(source, sink) and
       not isTestFile(sink.getNode().getLocation().getFile())
 select sink, source, sink,
   "CWE-537: Sensitive information flows into exception and is exposed via an error message."