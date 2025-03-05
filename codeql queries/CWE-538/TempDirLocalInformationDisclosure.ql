/**
 * @name Local information disclosure in a temporary directory with sensitive data
 * @kind path-problem
 * @id java/local-temp-file-sensitive-data-disclosure/538
 * @problem.severity error
 * @precision high
 * @tags security
 *      external/cwe/cwe-538
 * @cwe CWE-538
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.frameworks.Servlets
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.DataFlow
 import Barrier.Barrier
 import SensitiveInfo.SensitiveInfo


 module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;

 module TempFileFlow = TaintTracking::Global<TempFileConfig>;

 module TempFileConfig implements DataFlow::ConfigSig {
    predicate isSource(DataFlow::Node source) {
     exists(MethodCall getPropertyCall, NewClassExpr fileCreation |
        getPropertyCall.getMethod().hasQualifiedName("java.lang","System", "getProperty") and
        getPropertyCall.getArgument(0).(StringLiteral).getValue() = "java.io.tmpdir" and
        DataFlow::localExprFlow(getPropertyCall, fileCreation.getArgument(0)) and
        fileCreation.getConstructedType().hasQualifiedName("java.io", "File") and
        source.asExpr() = fileCreation
     )
    }

    predicate isSink(DataFlow::Node sink) {
      exists(MethodCall mc |
        mc.getMethod().hasName("write") and
        mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "Writer") and
        sink.asExpr() = mc.getQualifier())
    }
 }
 
 import Flow::PathGraph
 
 module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
    exists(Variable v | 
      // Use contains() with % wildcards instead of regex
      v.getName().matches("%oauthToken%") and  
      source.asExpr() = v.getAnAccess()
    )
    or
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)

  }
   predicate isSink(DataFlow::Node sink) {
    exists(MethodCall mc |
      mc.getMethod().hasName("write") and
      mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "Writer") and
      sink.asExpr() = mc.getAnArgument() and
      TempFileFlow::flowToExpr(mc.getQualifier())
    )
   }
 
  predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
   }
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink,
   "CWE-538: Insertion of Sensitive Information into Externally-Accessible File or Directory"