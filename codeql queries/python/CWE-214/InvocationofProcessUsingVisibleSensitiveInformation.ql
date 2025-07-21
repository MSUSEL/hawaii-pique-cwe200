/**
 * @name CWE-214: Invocation of Process Using Visible Sensitive Information
 * @description Detects when sensitive information is passed to a process execution command.
 * @kind path-problem
 * @problem.severity error
 * @precision high
 * @id python/invocation-of-process-using-visible-sensitive-information/214
 * @tags security
 *       external/cwe/cwe-200
 * @cwe CWE-214
 */

 import python
 import semmle.python.dataflow.new.DataFlow
 import semmle.python.dataflow.new.TaintTracking
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<ProcessExecutionWithSensitiveInfoConfig>;
 import Flow::PathGraph
 
 module ProcessExecutionWithSensitiveInfoConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   predicate isSink(DataFlow::Node sink) {
     // subprocess module calls
     exists(Call call |
       (
         call.getFunc().(Attribute).getName() in ["run", "call", "check_call", "check_output", "Popen"] and
         call.getFunc().(Attribute).getObject().(Name).getId() = "subprocess"
       ) and
       sink.asExpr() = call.getAnArg()
     )
     or
     // os.system() calls
     exists(Call call |
       call.getFunc().(Attribute).getName() = "system" and
       call.getFunc().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = call.getAnArg()
     )
     or
     // os.spawn* family of functions
     exists(Call call |
       call.getFunc().(Attribute).getName().matches("spawn%") and
       call.getFunc().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = call.getAnArg()
     )
     or
     // os.exec* family of functions
     exists(Call call |
       call.getFunc().(Attribute).getName().matches("exec%") and
       call.getFunc().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = call.getAnArg()
     )
     or
     // os.popen() calls
     exists(Call call |
       call.getFunc().(Attribute).getName() = "popen" and
       call.getFunc().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = call.getAnArg()
     )
     or
     // shlex.split() when used with process execution
     exists(Call call |
       call.getFunc().(Attribute).getName() = "split" and
       call.getFunc().(Attribute).getObject().(Name).getId() = "shlex" and
       sink.asExpr() = call.getAnArg()
     )
     or
     // Environment variable setting (os.environ, os.putenv)
     exists(Call call |
       (
         call.getFunc().(Attribute).getName() = "putenv" and
         call.getFunc().(Attribute).getObject().(Name).getId() = "os"
       ) and
       sink.asExpr() = call.getAnArg()
     )
     or
     // Environment variable assignment via os.environ
     exists(Subscript subscr |
       subscr.getObject().(Attribute).getName() = "environ" and
       subscr.getObject().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = subscr.getIndex()
     )
     or
     // Environment variable assignment via os.environ (assignment target)
     exists(Assign assign, Subscript subscr |
       assign.getATarget() = subscr and
       subscr.getObject().(Attribute).getName() = "environ" and
       subscr.getObject().(Attribute).getObject().(Name).getId() = "os" and
       sink.asExpr() = assign.getValue()
     )
     or
     // SSH/remote execution libraries (paramiko, fabric, etc.)
     exists(Call call |
       (
         call.getFunc().(Attribute).getName() in ["exec_command", "run", "sudo"] or
         call.getFunc().(Name).getId() in ["exec_command", "run", "sudo"]
       ) and
       sink.asExpr() = call.getAnArg()
     )
     or
     // Custom command execution sinks from configuration
     getSink(sink, "IPC Sink") 
    //  or
    //  getSink(sink, "Command Execution Sink")
   }
 
   predicate isBarrier(DataFlow::Node node) { 
     // Add Python-specific barriers if needed
     none()
   }
 }

 predicate isTestFile(File f) {
  // Convert path to lowercase for case-insensitive matching
  exists(string path | path = f.getAbsolutePath().toLowerCase() |
    // Check for common test-related directory or file name patterns
    path.regexpMatch(".*(test|tests|testing|test_suite|testcase|unittest|integration_test|spec).*")
  )
}
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 and not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink, "Sensitive information passed to process execution."
 