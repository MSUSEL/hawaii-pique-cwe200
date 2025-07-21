/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id python/error-message-sensitive-info/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */

 import python
 import semmle.python.dataflow.new.TaintTracking
 import semmle.python.dataflow.new.DataFlow
 
 import SensitiveInfo.SensitiveInfo
 import CommonSinks.CommonSinks
 
 private newtype MyFlowState =
   State1() or
   State2() or
   State3()
 
 module SensitiveInfoInErrorMsgConfig implements DataFlow::StateConfigSig {
 
   // Tie the module's FlowState to the newtype
   class FlowState = MyFlowState;
 
   // Source is valid only in State1
   predicate isSource(DataFlow::Node source, FlowState state) {
     state instanceof State1 and
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Sink is valid only in State3
   predicate isSink(DataFlow::Node sink, FlowState state) {
     state instanceof State3 and
     (
       // Common sinks from CommonSinks module
       CommonSinks::isPrintSink(sink) or
       CommonSinks::isErrPrintSink(sink) or
       CommonSinks::isWebFrameworkSink(sink) or
       CommonSinks::isErrorSink(sink) or
       // Custom sinks from configuration
       getSink(sink, "Log Sink") or
       getSink(sink, "Print Sink")
     )
   }
 
   // Transitions between states
   predicate isAdditionalFlowStep(
     DataFlow::Node node1, FlowState state1,
     DataFlow::Node node2, FlowState state2
   ) {
     // Transition from State1 to State2: 
     // Sensitive data passed into a NON-RUNTIME exception constructor
     (
       state1 instanceof State1 and
       state2 instanceof State2 and
       exists(Call call |
         // Exception constructor call - Accept ALL exceptions, then exclude runtime ones
         exists(string exceptionName |
           (
             call.getFunc().(Name).getId() = exceptionName or
             call.getFunc().(Attribute).getName() = exceptionName
           ) and
           // Exclude runtime exceptions (programming errors)
           not exceptionName in [
             "RuntimeError",      // Generic runtime error
             "ValueError",        // Invalid value (programming error)
             "TypeError",         // Type mismatch (programming error)
             "AttributeError",    // Missing attribute (programming error)
             "NameError",         // Undefined name (programming error)
             "IndexError",        // List index out of range (programming error)
             "KeyError",          // Dictionary key missing (programming error)
             "ZeroDivisionError", // Division by zero (programming error)
             "AssertionError",    // Failed assertion (programming error)
             "NotImplementedError", // Not implemented (programming error)
             "RecursionError",    // Too much recursion (programming error)
             "MemoryError",       // Out of memory (system error)
             "SystemError",       // Internal system error
             "ArithmeticError",   // Base for math errors (programming errors)
             "LookupError",       // Base for lookup errors (programming errors)
             "BufferError",       // Buffer operation error (programming error)
             "UnicodeError",      // Unicode encoding/decoding error (often programming error)
             "StopIteration",     // Iterator exhausted (programming logic)
             "GeneratorExit"      // Generator cleanup (programming logic)
           ]
         ) and
         call.getAnArg() = node1.asExpr() and
         call = node2.asExpr()
       )
     )
     or
     // Transition from State2 to State3: 
     // A 'raise' to a 'except' accessing the exception message
     (
       state1 instanceof State2 and
       state2 instanceof State3 and
       (
         // Exception object access in except block
         exists(Raise raise, ExceptStmt except |
           raise.getException() = node1.asExpr() and
           except.getName() = node2.asExpr()
         )
         or
         // String conversion of exception (str(e))
         exists(Call strCall |
           strCall.getFunc().(Name).getId() = "str" and
           strCall.getAnArg() = node1.asExpr() and
           strCall = node2.asExpr()
         )
         or
         // Exception message access via args attribute
         exists(Attribute attr |
           attr.getName() = "args" and
           attr.getObject() = node1.asExpr() and
           attr = node2.asExpr()
         )
       )
     )
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
 
 // Instantiate a GlobalWithState for taint tracking
 module SensitiveInfoInErrorMsgFlow = 
     TaintTracking::GlobalWithState<SensitiveInfoInErrorMsgConfig>;
 
 // Import its PathGraph (not the old DataFlow::PathGraph)
 import SensitiveInfoInErrorMsgFlow::PathGraph
 
 // Use the flowPath from that module
 from SensitiveInfoInErrorMsgFlow::PathNode source, SensitiveInfoInErrorMsgFlow::PathNode sink
 where SensitiveInfoInErrorMsgFlow::flowPath(source, sink) and
 not isTestFile(sink.getNode().getLocation().getFile())
 select sink, source, sink,
   "CWE-209: Sensitive information flows into exception and is exposed via an error message."
 