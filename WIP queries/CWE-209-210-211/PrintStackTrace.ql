/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in stack traces, potentially leading to information exposure.
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
 import Barrier.Barrier
 
 // Define flow states
//  class State1 extends DataFlow::FlowState { State1() { this = "State1" } }
//  class State2 extends DataFlow::FlowState { State2() { this = "State2" } }
//  class State3 extends DataFlow::FlowState { State3() { this = "State3" } }
 
 // Dataflow configuration
 class SensitiveInfoInStackTraceConfig extends TaintTracking::Configuration {
     SensitiveInfoInStackTraceConfig() { this = "SensitiveInfoInStackTraceConfig" }
 
     // Track sensitive variables as the source in State1
     override predicate isSource(DataFlow::Node source) {
         exists(SensitiveVariableExpr sve | source.asExpr() = sve)
     }
 
     // Track sinks as printStackTrace calls in State3
     override predicate isSink(DataFlow::Node sink) {
        //  state instanceof State3 and
         (
         exists(MethodCall mcSink |
             mcSink.getMethod().getName() = "printStackTrace" and
             sink.asExpr() = mcSink.getQualifier()
         ) 
         
        //  or
        //  exists(MethodCall mc |
        //     // Targets PrintWriter methods that may leak information
        //     mc.getMethod().hasName("println") 
        //     and
        //     // mc.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
        //     sink.asExpr() = mc.getAnArgument()
        //  )
         )
     }
 
    //  override predicate isAdditionalTaintStep(
    //     DataFlow::Node node1, DataFlow::FlowState state1,
    //     DataFlow::Node node2, DataFlow::FlowState state2
    //   ) {
    //     // Transition from State1 to State2: sensitive data flows into a runtime exception constructor
    //     state1 instanceof State1 and
    //     state2 instanceof State2 and
    //     exists(ConstructorCall cc |
    //       cc.getAnArgument() = node1.asExpr() and
    //       cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "Throwable") and
    //       // Exclude RuntimeException and its subclasses
    //       not cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("java.lang", "RuntimeException") and
    //       // Exclude ServletException and its subclasses
    //       not cc.getConstructor().getDeclaringType().(RefType).getASupertype+().hasQualifiedName("javax.servlet", "ServletException") and
    //       cc = node2.asExpr()
    //     ) or

    //       // Transition from State2 to State3: link throw to catch for getMessage()
    // state1 instanceof State2 and
    // state2 instanceof State3 and
    // exists(ThrowStmt t, CatchClause catchClause, MethodCall mcGetMessage |
    //   t.getExpr() = node1.asExpr() and
    //   catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
    //   catchClause.getVariable().getAnAccess() = mcGetMessage.getQualifier() and
    //   mcGetMessage.getMethod().getName() = "getMessage" and
    //   node2.asExpr() = mcGetMessage
    // ) or
    
    
    //     // Handle cases where the exception object itself (e) is passed to a method like print(e)
    //     state1 instanceof State2 and
    //     state2 instanceof State3 and
    //     exists(ThrowStmt t, CatchClause catchClause |
    //       t.getExpr() = node1.asExpr() and
    //       catchClause.getEnclosingCallable() = t.getEnclosingCallable() and
    //       node2.asExpr() = catchClause.getVariable().getAnAccess()
    //     )
    //   }
 
     override predicate isSanitizer(DataFlow::Node node) {
         Barrier::barrier(node)
     }
 }
 
 // Query for sensitive information flow from source to sink
 from DataFlow::PathNode source, DataFlow::PathNode sink, SensitiveInfoInStackTraceConfig cfg
 where cfg.hasFlowPath(source, sink)
 select sink, source, sink, "Sensitive information flows into printStackTrace."
 