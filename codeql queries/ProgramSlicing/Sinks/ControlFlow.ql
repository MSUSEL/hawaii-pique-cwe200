import java
import semmle.code.java.ControlFlowGraph
import SensitiveInfo.SensitiveInfo

predicate controlFlowPath(MethodCall start, MethodCall end) {
  exists(ControlFlowNode startNode, ControlFlowNode endNode |
    startNode = start.getControlFlowNode() and
    endNode = end.getControlFlowNode() and
    startNode.getANormalSuccessor*() = endNode // Explore control flow graph for normal successors
  )
}

from MethodCall normalCall, DetectedMethodCall sensitiveCall
where controlFlowPath(normalCall, sensitiveCall) 
select normalCall, sensitiveCall, "Control flow path from normal method call to sensitive method call."
