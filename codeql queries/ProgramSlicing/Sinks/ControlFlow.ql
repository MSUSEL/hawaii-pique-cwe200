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
     or
     // Include cast expressions where sensitive data might flow between different types
     source.asExpr() instanceof CastExpr
   }
 
   predicate isSink(DataFlow::Node sink) {
    exists(DetectedMethodCall dmc |
      sink.asExpr() = dmc.getAnArgument()
    )
  }
  

 
   // Optimize additional flow steps by capturing key flow connections but limiting unnecessary links
   predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    // Flow from arguments to method calls
    exists(MethodCall call |
      pred.asExpr() = call.getArgument(_) and
      succ.asExpr() = call
    )
    or
    // Flow through binary expressions (e.g., string concatenations)
    exists(BinaryExpr binExpr |
      pred.asExpr() = binExpr.getAnOperand() and
      succ.asExpr() = binExpr
    )
    // or
    // Existing steps (if any)
    // ...
  }
}
 
 from Flow::PathNode source, Flow::PathNode sink, Method finalCallee
 where
   // Capture flows between optimized sources and sinks
   Flow::flowPath(source, sink) and
   exists(MethodCall mc | sink.getNode().asExpr() = mc.getAnArgument() | finalCallee = mc.getCallee())

    
  select
   sink.getNode().getEnclosingCallable(),
   source,
   sink,
   "Dataflow from $@ || $@",
   source,
   source.toString(),
   finalCallee,
   finalCallee.toString()