/**
 * @name CWE-532: Log-Sensitive-Information
 * @description Writing sensitive information to log files can allow that
 *              information to be leaked to an attacker more easily.
 * @kind path-problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision medium
 * @id java/log-sensitive-information/532
 * @tags security
 *       external/cwe/cwe-532
 * @cwe CWE-532
 */

 import java
 private import semmle.code.java.dataflow.ExternalFlow
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.frameworks.android.Compose
 private import semmle.code.java.security.Sanitizers
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module PermissionBarrier {
   predicate isPermissionCheck(MethodCall ma) {
     ma.getMethod().getName().matches("%Permission%") or
     ma.getMethod().getName().matches("%authorize%") or
     ma.getMethod().getName().matches("%canAccess%") or
     ma.getMethod().getName().matches("%isAuthorized%")
   }
 
   predicate isPermissionBarrier(DataFlow::Node node) {
     exists(MethodCall ma |
       isPermissionCheck(ma) and
       node.asExpr() = ma.getQualifier()
     )
   }
 }
 
 private class TypeType extends RefType {
   pragma[nomagic]
   TypeType() {
     this.getSourceDeclaration().getASourceSupertype*().hasQualifiedName("java.lang.reflect", "Type")
   }
 }
 
 /**
  * Predicate to determine if a file is test-related based on its full path
  */
 predicate isTestFile(File f) {
   // Convert path to lowercase for case-insensitive matching
   exists(string path | path = f.getAbsolutePath().toLowerCase() |
     // Check for common test-related directory or file name patterns
     path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
   )
 }
 
 module SensitiveLoggerConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) { 
     exists(Variable v | 
       (
         (v.getName().toLowerCase().regexpMatch(".*secret.*") or
          v.getName().toLowerCase().regexpMatch(".*credential.*") or
          v.getName().toLowerCase().regexpMatch(".*passw.*") or
          v.getName().toLowerCase().regexpMatch(".*pwd.*") or
          v.getName().toLowerCase().regexpMatch("pin")
         )
         and
         not v.getName().toLowerCase().regexpMatch(".*id.*")
       )
       and  
       source.asExpr() = v.getAnAccess()
     )
   }
   
   predicate isSink(DataFlow::Node sink) { 
     sinkNode(sink, "log-injection") 
   }
 
   predicate isBarrier(DataFlow::Node sanitizer) {
     sanitizer.asExpr() instanceof LiveLiteral or
     sanitizer instanceof SimpleTypeSanitizer or
     sanitizer.getType() instanceof TypeType or
     PermissionBarrier::isPermissionBarrier(sanitizer) or
     Barrier::barrier(sanitizer)
   }
 
   predicate isBarrierIn(DataFlow::Node node) { 
     isSource(node) 
   }
 }
 
 module SensitiveLoggerFlow = TaintTracking::Global<SensitiveLoggerConfig>;
 import SensitiveLoggerFlow::PathGraph
 
 from SensitiveLoggerFlow::PathNode source, SensitiveLoggerFlow::PathNode sink
 where SensitiveLoggerFlow::flowPath(source, sink) 
   and not exists(MethodCall ma |
     PermissionBarrier::isPermissionCheck(ma) and
     ma.getEnclosingCallable() = sink.getNode().getEnclosingCallable()
   )
   // Exclude test files based on the full path of the sink
   and not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink, "This $@ is written to a log file.", source.getNode(),
   "potentially sensitive information"