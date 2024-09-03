/**
 * @name Backward slicing
 * @description Identifies the backward slice of a sink node
 * @kind path-problem
 * @problem.severity warning
 * @id java/backward-slice
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
 
  predicate isSource(DataFlow::Node source) {
    // Include sensitive variables as sources
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
    or
    // Include literals and constants as sources
    source.asExpr() instanceof Literal
    or
    // Include variables initialized directly with literals, function calls, field accesses, or new object instantiations
    exists(Variable v |
      source.asExpr() = v.getAnAccess() and
      (
        v.getInitializer() instanceof MethodCall or
        v.getInitializer() instanceof FieldAccess or
        v.getInitializer() instanceof NewClassExpr
      )
    )
    or
    // Include direct function/method calls as sources
    source.asExpr() instanceof MethodCall
    or
    // Include new object instantiations as sources
    source.asExpr() instanceof NewClassExpr
    or
    // Include field accesses as sources
    source.asExpr() instanceof FieldAccess
    or
    // Include method parameters as sources
    exists(Parameter p |
      source.asExpr() = p.getAnAccess()
    )
    or
    // Include constructor arguments as sources
    exists(NewClassExpr newExpr |
      newExpr.getAnArgument() = source.asExpr()
    )
  }
 
   predicate isSink(DataFlow::Node sink) {
     exists(SensitiveVariableExpr sve | sink.asExpr() = sve)
   }
 
  //  predicate neverSkip(DataFlow::Node node) {
  //    any()
  //  }
   
   predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    exists(MethodCall call |
      pred.asExpr() = call.getArgument(_) and
      succ.asExpr() = call
    )  // Add flow from constructor arguments to the instantiated object
    or exists(NewClassExpr newExpr |
      pred.asExpr() = newExpr.getArgument(_) and
      succ.asExpr() = newExpr
    )        // Capture flow from method call expressions to variable assignments
   //  or exists(MethodCall call, VarAccess va |
   //    pred.asExpr() = call and
   //    succ.asExpr() = va
   //  )
  //  any()
  }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where
   // Capture flows between source and sink
   Flow::flowPath(source, sink)
 select
   sink.getNode(),
   source,
   sink,
   source.getNode().getType().toString(),
   sink.getNode().getType().toString(),
   "Dataflow from `" + source.getNode().toString() + "` to `" + sink.getNode().toString() + "`"
 