/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information (Direct)
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id python/error-message-sensitive-info-direct/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */


 import python
 import semmle.python.dataflow.new.TaintTracking
 import semmle.python.dataflow.new.DataFlow
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo

 
 module Flow = TaintTracking::Global<GenerationOfErrorMessageWithSensInfoConfig>;
 
 import Flow::PathGraph
 
 module GenerationOfErrorMessageWithSensInfoConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }
 
  
   predicate isSink(DataFlow::Node sink) {
     // Consider the case where the sink exposes sensitive info within an except clause for NON-RUNTIME exceptions
     exists(Call call, ExceptStmt except |
       // Check if the except clause handles non-runtime exceptions
       // Accept ALL exceptions, then exclude runtime ones
       (
         // Handle cases where except has a specific type
         exists(string exceptionName |
           except.getType().(Name).getId() = exceptionName and
           // Exclude runtime exceptions (programming errors)
           not exceptionName in [
             "RuntimeError", "ValueError", "TypeError", "AttributeError", "NameError",
             "IndexError", "KeyError", "ZeroDivisionError", "AssertionError", "NotImplementedError",
             "RecursionError", "MemoryError", "SystemError", "ArithmeticError", "LookupError",
             "BufferError", "UnicodeError", "StopIteration", "GeneratorExit"
           ]
         )
         or
         // Handle bare except: clauses (catches all exceptions)
         not exists(except.getType())
       ) and
       // The call is within the except block
       exists(Stmt stmt |
         stmt = except.getBody().getAnItem() and
         call.getParent*() = stmt
       ) and
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrPrintSink(sink) or
         CommonSinks::isLoggingSink(sink) or
         CommonSinks::isErrorSink(sink) or
         CommonSinks::isWebFrameworkSink(sink) or
         CommonSinks::isPythonFrameworkSink(sink) or
         // Custom sinks
         getSink(sink, "Log Sink") or
         getSink(sink, "Print Sink")
       ) and
       sink.asExpr() = call.getAnArg()
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
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink) and
 not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink,
   "CWE-209: Sensitive information included in error messages."