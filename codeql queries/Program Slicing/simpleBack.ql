/**
 * @name Backward slicing
 * @description identifies the backward slice of a sink node
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
  // predicate isSource(DataFlow::Node source) {
  //   exists(Variable v |
  //     (
  //       v.getName() = "a" or
  //       v.getName() = "b" or
  //       v.getName() = "c" or
  //       v.getName() = "d" or
  //       v.getName() = "e" or
  //       v.getName() = "f" or
  //       v.getName() = "g" 
  //     ) and
  //     source.asExpr() = v.getAnAccess()
  //   )
  // }


  predicate isSource(DataFlow::Node source) {
    // Include variables marked as sensitive
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
    or
    // Include literals and constants as sources
    source.asExpr() instanceof Literal
    or
    source.asExpr() instanceof FieldAccess
    or
    // Include variables that are initialized directly with literals, function calls, or field accesses
    exists(Variable v |
      source.asExpr() = v.getAnAccess() and
      (
        v.getInitializer() instanceof MethodCall or
        v.getInitializer() instanceof FieldAccess
      )
    )
    or
    // Include direct function/method calls as sources
    source.asExpr() instanceof MethodCall
    or
    // Include field accesses as sources
    source.asExpr() instanceof FieldAccess
  }


  predicate isSink(DataFlow::Node sink) {
    exists(Variable v |
      (
        v.getName() = "a" or
        v.getName() = "b" or
        v.getName() = "c" or
        v.getName() = "d" or
        v.getName() = "e" or
        v.getName() = "f" or
        v.getName() = "g"
      ) and
      sink.asExpr() = v.getAnAccess()
    )
  }

  predicate neverSkip(DataFlow::Node node) { any() }


}

from Flow::PathNode source, Flow::PathNode sink
where
  Flow::flowPath(source, sink) and
  // Exclude cases where the source and sink are the same node
  source.getNode().toString() != sink.getNode().toString()
select sink.getNode(), source, sink,
source.getNode().getType().toString(),
sink.getNode().getType().toString(),
  "Dataflow from `" + source.getNode().toString() + "` to `" + sink.getNode().toString()
