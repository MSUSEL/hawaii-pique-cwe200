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
 
 module Flow = TaintTracking::Global<AllVariablesConfig>;
 import Flow::PathGraph
 
 /** A configuration for tracking data flow between all variables. */
 module AllVariablesConfig implements DataFlow::ConfigSig {
 
   // Any variable or method call argument can be a source
   predicate isSource(DataFlow::Node source) {
     source.asExpr() instanceof Literal
     or
     exists(Variable v |
       source.asExpr() = v.getAnAccess()
     )
     or
     source.asExpr() instanceof MethodCall
     or
     source.asExpr() instanceof NewClassExpr
     or
     source.asExpr() instanceof FieldAccess
     or
     exists(Parameter p |
       source.asExpr() = p.getAnAccess()
     )
   }
 
   // Any variable or method call return value can be a sink
   predicate isSink(DataFlow::Node sink) {
     exists(Variable v |
       sink.asExpr() = v.getAnAccess()
     )
   }
 
   // Do not skip any nodes
   predicate neverSkip(DataFlow::Node node) {
     any()
   }
 
   /** Additional flow steps to track flow from method arguments to return values */
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
    // any()
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink) and
     // Exclude cases where the source and sink are the same node
     source.getNode().toString() != sink.getNode().toString()
 select sink.getNode(), source, sink, 
    "Dataflow from `" + source.getNode().toString() + "` to `" + sink.getNode().toString() + "`"
 