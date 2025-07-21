/**
 * @name CWE-214: Invocation of Process Using Visible Sensitive Information
 * @description Detects when sensitive information is passed to a process execution command.
 * @kind path-problem
 * @problem.severity error
 * @precision high
 * @id java/invocation-of-process-using-visible-sensitive-information/214
 * @tags security
 *       external/cwe/cwe-200
 * @cwe CWE-214
 */

 import java
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.TaintTracking
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<ProcessExecutionWithSensitiveInfoConfig>;
 import Flow::PathGraph
 
 module ProcessExecutionWithSensitiveInfoConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   predicate isSink(DataFlow::Node sink) {
     // Checks if the sink is a method call to Runtime.exec
     exists(MethodCall execCall |
       execCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
       execCall.getMethod().hasName("exec") and
       sink.asExpr() = execCall.getAnArgument()
     )
     or
     // Checks if the sink is a method call to ProcessBuilder
     exists(MethodCall envCall |
       envCall.getMethod().hasName("ProcessBuilder") and
       sink.asExpr() = envCall.getQualifier()
     )
     or
     // Checks if the sink is a method call to ProcessBuilder.environment.put
     exists(MethodCall putCall, MethodCall envCall |
       putCall.getMethod().hasName("put") and
       envCall = putCall.getQualifier().(MethodCall) and
       envCall.getMethod().hasName("environment") and
       envCall.getQualifier().getType().(RefType).hasQualifiedName("java.lang", "ProcessBuilder") and
       sink.asExpr() = putCall.getAnArgument()
     )
     or
       // ProcessBuilder.command(String...) or ProcessBuilder.command(List<String>)
     exists(MethodCall cmdCall |
     cmdCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "ProcessBuilder") and
     cmdCall.getMethod().hasName("command") and
     sink.asExpr() = cmdCall.getAnArgument()
   )
     or
     // ProcessBuilder constructor with arguments
     exists(MethodCall ctorCall |
       ctorCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "ProcessBuilder") and
       ctorCall.getMethod().getName() = "ProcessBuilder" and
       sink.asExpr() = ctorCall.getAnArgument()
     )
     or
     // Apache Commons Exec CommandLine sinks
     exists(MethodCall cmdCall |
       cmdCall
           .getMethod()
           .getDeclaringType()
           .hasQualifiedName("org.apache.commons.exec", "CommandLine") and
       (
         cmdCall.getMethod().hasName("addArgument") or
         cmdCall.getMethod().hasName("addArguments")
       ) and
       sink.asExpr() = cmdCall.getAnArgument()
     )
     or
     // JSch Session.execCommand sink
     exists(MethodCall execCall |
       execCall.getMethod().getDeclaringType().hasQualifiedName("com.jcraft.jsch", "Session") and
       execCall.getMethod().hasName("execCommand") and
       sink.asExpr() = execCall.getAnArgument()
     )
     or
     // Use the LLM response to indentify command execution sinks
     getSink(sink, "IPC Sink")
   }
 
   predicate isBarrier(DataFlow::Node node) { Barrier::barrier(node) }
 }

 predicate isTestFile(File f) {
  // Convert path to lowercase for case-insensitive matching
  exists(string path | path = f.getAbsolutePath().toLowerCase() |
    // Check for common test-related directory or file name patterns
    path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
  )
}
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 and not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink, "Sensitive information passed to process execution."
 