/**
 * @name Insertion of Sensitive Information into Externally-Accessible File
 * @description Writing sensitive information to log files can allow that
 *              information to be leaked to an attacker more easily.
 * @kind path-problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision medium
 * @id java/sensitive-info-log-file/538
 * @tags security
 *       cwe-538
 * @cwe CWE-538
 */

 import java
 import shared.SensitiveVariables
 import shared.SensitiveLoger
 import SensitiveLoggerFlow::PathGraph
 
 from SensitiveLoggerFlow::PathNode source, SensitiveLoggerFlow::PathNode sink
 where SensitiveLoggerFlow::flowPath(source, sink)
 select sink.getNode(), source, sink, "This $@ is written to a log file.", source.getNode(),
   "potentially sensitive information"


