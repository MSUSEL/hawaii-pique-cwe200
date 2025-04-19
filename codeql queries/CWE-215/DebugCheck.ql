/**
 * @name CWE-215: Sensitive data exposure in debug code
 * @description Identifies instances where sensitive variables are exposed in debug output without proper sanitization, which could lead to sensitive data leaks.
 * @kind path-problem
 * @problem.severity warning
 * @id java/debug-code-sensitive-data-exposure/215
 * @tags security
 *      external/cwe/cwe-215
 * @cwe CWE-215
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<SensitiveDataConfig>;
 import Flow::PathGraph
 
 // Define sensitive data sources (e.g., sensitive variables)
 module SensitiveDataConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Define debug output sinks (e.g., print statements or logging methods)
   predicate isSink(DataFlow::Node sink) {
     getSinkAny(sink)
   }
 
   // Temporarily disable barriers for debugging
   predicate isBarrier(DataFlow::Node node) {
     none() // Barrier::barrier(node)
   }
 }
 
 // Identify Debug Flag Checks
 class DebugFlagCheck extends IfStmt {
  DebugFlagCheck() {
    exists(VarAccess va |
      va = this.getCondition() and
      va.getVariable().getName().toLowerCase().regexpMatch(".*debug.*")
    )
  }
 
   // Check if a given expression is in the 'then' branch or a method called from it
   predicate isInThenBranch(Expr e) {
     (
       // Direct inclusion in the then branch
       e.getEnclosingStmt().getParent*() = this.getThen()
     ) or
     (
       // Expression in a method called from the then branch
       exists(MethodCall mc, Method calledMethod |
         mc.getEnclosingStmt().getParent*() = this.getThen() and
         calledMethod = mc.getCallee() and
         e.getEnclosingCallable() = calledMethod
       )
     )
   }
 
   // Ensure no sanitization method is in the same method scope as the debug check
   predicate noSanitizationInScope() {
     not exists(MethodCall mc |
       mc.getMethod().getName().toLowerCase().matches("%sanitize%") and
       mc.getEnclosingCallable() = this.getEnclosingCallable()
     )
   }
 }
 
 // Handle flows through exception fields
 class SensitiveExceptionFieldAccess extends MethodCall {
   SensitiveExceptionFieldAccess() {
     this.getMethod().getName().toLowerCase().regexpMatch("get.*") and
     this.getQualifier().getType().(RefType).getASupertype*().hasQualifiedName("java.lang", "Exception") and
     exists(DataFlow::Node source, DataFlow::Node sink |
       Flow::flow(source, sink) and
       sink.asExpr() = this.getQualifier()
     )
   }
 }
 
 // Query for sensitive data flow within debug flag's 'then' branch
 from Flow::PathNode source, Flow::PathNode sink, DebugFlagCheck dfc
 where
   Flow::flowPath(source, sink) and
   (
     dfc.isInThenBranch(sink.getNode().asExpr()) or
     // Handle exception field accesses
     exists(SensitiveExceptionFieldAccess sefa |
       dfc.isInThenBranch(sefa) and
       sink.getNode().asExpr() = sefa
     )
   ) and
   dfc.noSanitizationInScope()
 select sink.getNode(), source, sink, "Sensitive data exposed in debug code."