/**
 * @name Android Webview debugging enabled
 * @description Enabling Webview debugging in production builds can expose entry points or leak sensitive information.
 * @kind path-problem
 * @problem.severity warning
 * @security-severity 7.2
 * @id CWE-215
 * @tags security
 *       external/cwe/cwe-489
 * @precision high
 * @cwe CWE-215
 */

 import java
 import ShellErrorExposureFlowLib
//  import ShellErrorExposureFlow::PathGraph
 
 from ShellErrorExposureFlow::PathNode source, ShellErrorExposureFlow::PathNode sink
 where ShellErrorExposureFlow::flowPath(source, sink)
 select sink, source, sink, "Webview debugging is enabled."
 