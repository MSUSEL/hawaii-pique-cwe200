/**
 * @name CWE-532: Log-Sensitive-Information
 * @description Writing sensitive information to log files can allow that
 *              information to be leaked to an attacker more easily.
 * @kind path-problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision medium
 * @id python/log-sensitive-information/532
 * @tags security
 *       external/cwe/cwe-532
 * @cwe CWE-532
 */

 import python
 private import semmle.python.dataflow.new.DataFlow
 import semmle.python.dataflow.new.TaintTracking
 import SensitiveInfo.SensitiveInfo
 
 module PermissionBarrier {
   predicate isPermissionCheck(Call call) {
     exists(string funcName |
       (
         call.getFunc().(Name).getId() = funcName or
         call.getFunc().(Attribute).getName() = funcName
       ) and
       (
         funcName.matches("%permission%") or
         funcName.matches("%authorize%") or
         funcName.matches("%can_access%") or
         funcName.matches("%is_authorized%")
       )
     )
   }
 
   predicate isPermissionBarrier(DataFlow::Node node) {
     exists(Call call |
       isPermissionCheck(call) and
       node.asExpr() = call.getFunc()
     )
   }
 }
 
 /**
  * Predicate to determine if a file is test-related based on its full path
  */
 predicate isTestFile(File f) {
   // Convert path to lowercase for case-insensitive matching
   exists(string path | path = f.getAbsolutePath().toLowerCase() |
     // Check for common test-related directory or file name patterns
     path.regexpMatch(".*(test|tests|testing|test_suite|testcase|unittest|integration_test|spec).*")
   )
 }
 
 module SensitiveLoggerConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) { 
     exists(Name name | 
       (
         (name.getId().toLowerCase().regexpMatch(".*secret.*") or
          name.getId().toLowerCase().regexpMatch(".*credential.*") or
          name.getId().toLowerCase().regexpMatch(".*passw.*") or
          name.getId().toLowerCase().regexpMatch(".*pwd.*") or
          name.getId().toLowerCase().regexpMatch("pin") or
          name.getId().toLowerCase().regexpMatch(".*apikey.*") or
          name.getId().toLowerCase().regexpMatch("ssn") or
          name.getId().toLowerCase().regexpMatch("decryptionkey") or
          name.getId().toLowerCase().regexpMatch("privatekey") or
          name.getId().toLowerCase().regexpMatch("creditcardnumber") or
          name.getId().toLowerCase().regexpMatch("cvv*") or
          name.getId().toLowerCase().regexpMatch(".+token$")         )
         and
         not name.getId().toLowerCase().regexpMatch(".*id.*")
       )

      and  
      source.asExpr() = name
     )
   }
   
   predicate isSink(DataFlow::Node sink) { 
     getSink(sink, "Log Sink") or
     // Common Python logging functions
     exists(Call call |
       (
         call.getFunc().(Name).getId() in ["print", "log", "debug", "info", "warning", "error", "critical"] or
         call.getFunc().(Attribute).getName() in ["print", "log", "debug", "info", "warning", "error", "critical", "write"]
       ) and
       sink.asExpr() = call.getAnArg()
     )
   }
 
   predicate isBarrier(DataFlow::Node sanitizer) {
     PermissionBarrier::isPermissionBarrier(sanitizer)
   }
 
   predicate isBarrierIn(DataFlow::Node node) { 
     isSource(node) 
   }
 }
 
 module SensitiveLoggerFlow = TaintTracking::Global<SensitiveLoggerConfig>;
 import SensitiveLoggerFlow::PathGraph
 
 from SensitiveLoggerFlow::PathNode source, SensitiveLoggerFlow::PathNode sink
 where SensitiveLoggerFlow::flowPath(source, sink) 
   and not exists(Call call |
     PermissionBarrier::isPermissionCheck(call) and
     call.getScope() = sink.getNode().getScope()
   )
   // Exclude test files based on the full path of the sink
   and not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink, "CWE-532: Sensitive information (" + source.getNode() +") written to a log file."