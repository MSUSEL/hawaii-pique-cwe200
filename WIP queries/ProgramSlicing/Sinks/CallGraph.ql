/**
 * @kind path-problem
 * @id java/sinks/call-graph
 */

 import java
 import SensitiveInfo.SensitiveInfo

 query predicate edges(Method a, Method b) { a.calls(b) }
 
 from DetectedMethod end, Method entryPoint
 where edges+(entryPoint, end)
 select entryPoint, entryPoint, end, "Found a path from start to target."
