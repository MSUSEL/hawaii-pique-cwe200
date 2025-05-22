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
 
 import SensitiveInfo.SensitiveInfo
 import CommonSinks.CommonSinks
 import Barrier.Barrier
 private import semmle.code.java.security.InformationLeak
 
 private newtype MyFlowState =
   State1() or
   State2() or
   State3()
 
 module SensitiveInfoInErrorMsgConfig implements DataFlow::StateConfigSig {
 
   class FlowState = MyFlowState;
 
   predicate isSource(DataFlow::Node source, FlowState state) {
     state instanceof State1 and
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   predicate isSink(DataFlow::Node sink, FlowState state) {
     state instanceof State3 and
     exists(MethodCall mcSink |
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isServletSink(sink) or
         sink instanceof InformationLeakSink
       ) and
       sink.asExpr() = mcSink.getAnArgument()
     )
   }
 
   predicate isAdditionalFlowStep(
     DataFlow::Node node1, FlowState state1,
     DataFlow::Node node2, FlowState state2
   ) {
     // Transition from State1 to State2: 
     // Sensitive data passed into a non-RuntimeException constructor
     (
       state1 instanceof State1 and
       state2 instanceof State2 and
       exists(ConstructorCall cc |
         cc.getAnArgument() = node1.asExpr() and
         cc.getConstructor().getDeclaringType().(RefType).getASupertype+()
           .hasQualifiedName("java.lang", "Throwable") and
         not cc.getConstructor().getDeclaringType().(RefType).getASupertype+()
           .hasQualifiedName("java.lang", "RuntimeException") and
         not cc.getConstructor().getDeclaringType().(RefType).getASupertype+()
           .hasQualifiedName("javax.servlet", "ServletException") and
         cc = node2.asExpr()
       )
     )
     or
     // Transition from State2 to State3: 
     // A 'throw' to a 'catch', including cross-method propagation
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       (
         // Case 1: Same method (existing logic)
         exists(
           ThrowStmt t,
           CatchClause catchClause,
           MethodCall mcGetMessage |
           t.getExpr() = node1.asExpr() and
           catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
           catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
           mcGetMessage.getMethod().getName() = "getMessage" and
           node2.asExpr() = mcGetMessage
         )
         or
         exists(
           ThrowStmt t,
           CatchClause catchClause |
           t.getExpr() = node1.asExpr() and
           catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
           node2.asExpr() = catchClause.getVariable().getAnAccess()
         )
         or
         // Case 2: Cross-method propagation (new logic)
         exists(
           ThrowStmt t,
           CatchClause catchClause,
           MethodCall mcGetMessage,
           Call call |
           t.getExpr() = node1.asExpr() and
           // The throw is in a method called by the caller
           call.getCallee() = t.getEnclosingCallable() and
           // The catch clause is in the caller's method
           catchClause.getEnclosingCallable() = call.getEnclosingCallable() and
           // The catch clause catches the thrown exception type
           catchClause.getACaughtType().getASupertype*() = t.getExpr().getType() and
           // The catch variable is used in a getMessage() call
           catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
           mcGetMessage.getMethod().getName() = "getMessage" and
           node2.asExpr() = mcGetMessage
         )
         or
         exists(
           ThrowStmt t,
           CatchClause catchClause,
           Call call |
           t.getExpr() = node1.asExpr() and
           // The throw is in a method called by the caller
           call.getCallee() = t.getEnclosingCallable() and
           // The catch clause is in the caller's method
           catchClause.getEnclosingCallable() = call.getEnclosingCallable() and
           // The catch clause catches the thrown exception type
           catchClause.getVariable().getType().(RefType).hasSubtype*(t.getExpr().getType()) and
           // The catch variable itself is used (e.g., passed to another method)
           node2.asExpr() = catchClause.getVariable().getAnAccess()
           )
       )
     )
     or
     // New transition: State3 to State3 for method calls
     // Tracks the flow from catch variable access to a method argument (e.g., logError)
     (
       state1 instanceof State3 and
       state2 instanceof State3 and
       exists(MethodCall mc |
         node1.asExpr() = mc.getQualifier() and
         mc.getMethod().getName() = "getMessage" and
         node2.asExpr() = mc.getAnArgument()
       )
       or
       exists(MethodCall mc, DataFlow::Node paramUse, Parameter param |
        state1 instanceof State3 and
        state2 instanceof State3 and
        node1.asExpr() = mc.getAnArgument() and
        mc.getCallee().getAParameter() = param and
        node2 = DataFlow::parameterNode(param) and
        // Ensure the parameter is used in the called method
        DataFlow::localFlowStep(node2, paramUse)
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
 
 module SensitiveInfoInErrorMsgFlow = TaintTracking::GlobalWithState<SensitiveInfoInErrorMsgConfig>;
 
 import SensitiveInfoInErrorMsgFlow::PathGraph
 
 from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
 where SensitiveInfoInErrorMsgFlow::flowPath(source, sink) and
       not isTestFile(sink.getNode().getLocation().getFile())
 select sink, source, sink,
   "CWE-209: Sensitive information flows into exception and is exposed via an error message."