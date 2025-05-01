/**
 * @name Backward slicing (optimized with extended flow tracking)
 * @description Identifies the backward slice of a sink node with performance improvements and additional tracking for transformations
 * @kind path-problem
 * @problem.severity warning
 * @id java/backward-slice-extended
 */

 import java
 private import semmle.code.java.dataflow.ExternalFlow
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<SensitiveInfoInErrorMsgConfig>;
 import Flow::PathGraph
 
 /** A configuration for tracking sensitive information flow into error messages. */
 module SensitiveInfoInErrorMsgConfig implements DataFlow::ConfigSig {
 
   // Optimized source predicate to reduce performance overhead and track additional types
   predicate isSource(DataFlow::Node source) {
     // Include only sensitive variables explicitly marked
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
     or
     // Include all string literals (including numeric-like strings)
     source.asExpr() instanceof StringLiteral
     or
     // Track method calls if they are assigned to variables (right-hand side of assignment)
     exists(AssignExpr assign |
       assign.getRhs() = source.asExpr() and
       source.asExpr() instanceof MethodCall
     )
     or
     // Track static method calls
     exists(AssignExpr assign |
       assign.getRhs() = source.asExpr() and
       source.asExpr() instanceof StaticMethodCall
     )
     or
     // Track field accesses if they are assigned to variables (right-hand side of assignment)
     exists(AssignExpr assign |
       assign.getRhs() = source.asExpr() and
       source.asExpr() instanceof FieldAccess
     )
     or
     // Include constructor arguments as sources to capture flow into objects
     exists(NewClassExpr newExpr |
       newExpr.getAnArgument() = source.asExpr()
     )
    //  or
    //  // Include cast expressions where sensitive data might flow between different types
    //  source.asExpr() instanceof CastExpr
   }
 
   // Sink predicate remains unchanged; all sensitive variables are considered sinks
   predicate isSink(DataFlow::Node sink) {
     exists(SensitiveVariableExpr sve | sink.asExpr() = sve)
   }
 
   // Optimize additional flow steps by capturing key flow connections but limiting unnecessary links
  //  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
  //    // Limit tracking from method arguments to method calls and constructor arguments to objects
  //    exists(MethodCall call |
  //      pred.asExpr() = call.getArgument(_) and
  //      succ.asExpr() = call
  //    )
  //    or
  //    exists(NewClassExpr newExpr |
  //      pred.asExpr() = newExpr.getArgument(_) and
  //      succ.asExpr() = newExpr
  //    )
  //    or
  //    // Capture flows involving cast expressions
  //    exists(CastExpr cast |
  //      pred.asExpr() = cast.getUnderlyingExpr() and
  //      succ.asExpr() = cast
  //    )
  //  }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where
   // Capture flows between optimized sources and sinks
   Flow::flowPath(source, sink)
 select
   sink.getNode(),
   source,
   sink,
   source.getNode().getType().toString(),
   sink.getNode().getType().toString(),
   "Dataflow from " + source.getNode().toString() + " to " + sink.getNode().toString() + ""