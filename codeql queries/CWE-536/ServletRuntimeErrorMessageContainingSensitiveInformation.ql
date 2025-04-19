/**
 * @name CWE-536: Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-runtime-error-message-exposure/536
 * @tags security
 *       external/cwe/cwe-536
 * @cwe CWE-536
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
 
 module ExceptionDataFlowConfig implements DataFlow::StateConfigSig {
   class FlowState = MyFlowState;
 
   // 1) Sensitive data as source
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
 
   // 4) Sinks: servlet-response, print, log
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
     // (1) Sensitive → new ServletException(...) → State2
     (
       state1 instanceof State1 and
       state2 instanceof State2 and
       exists(ConstructorCall cc, RefType t |
         cc.getAnArgument() = node1.asExpr() and
         t = cc.getConstructor().getDeclaringType().(RefType).getASupertype*() and
         (
           t.hasQualifiedName("javax.servlet", "ServletException") or
           t.hasQualifiedName("jakarta.servlet", "ServletException")
         ) and
         cc = node2.asExpr()
       )
     ) or
     // (2) Propagate across call boundary via MethodCall → State2
     (
       state1 instanceof State2 and
       state2 instanceof State2 and
       exists(MethodCall mcCall, RefType thrownType |
         mcCall = node2.asExpr() and
         thrownType = mcCall.getMethod().getAThrownExceptionType().(RefType) and
         (
           thrownType.hasQualifiedName("javax.servlet", "ServletException") or
           thrownType.hasQualifiedName("jakarta.servlet", "ServletException")
         )
       )
     ) or
     // (3) In-method ThrowStmt → catch via getMessage() → State3
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       exists(ThrowStmt t, CatchClause catchClause, MethodCall mcGetMessage |
         t.getExpr() = node1.asExpr() and
         catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
         catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
         mcGetMessage.getMethod().getName() = "getMessage" and
         node2.asExpr() = mcGetMessage
       )
     ) or
     // (3b) Cross-method ThrowStmt → catch via getMessage() → State3 (refined)
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
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
     ) or
     // (4) In-method ThrowStmt → catch via exception object → State3
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       exists(ThrowStmt t, CatchClause catchClause |
         t.getExpr() = node1.asExpr() and
         catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
         node2.asExpr() = catchClause.getVariable().getAnAccess()
       )
     ) or
     // (4b) Cross-method ThrowStmt → catch via exception object → State3 (refined)
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
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
     ) or
     // (5) State3 → State3: Parameter flow to expression use (new)
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
     // (6) State3 → State3: String concatenation (new)
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
   exists(string path |
     path = f.getAbsolutePath().toLowerCase() and
     path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
   )
 }
 
 module SensitiveInfoInErrorMsgFlow = TaintTracking::GlobalWithState<ExceptionDataFlowConfig>;
 
 import SensitiveInfoInErrorMsgFlow::PathGraph
 
 from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
 where
   SensitiveInfoInErrorMsgFlow::flowPath(source, sink) and
   not isTestFile(sink.getNode().getLocation().getFile())
 select
   sink, source, sink,
   "CWE-536: Sensitive information flows into a servlet runtime exception and is exposed via an error message."