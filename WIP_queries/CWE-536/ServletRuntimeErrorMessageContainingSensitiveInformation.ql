/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/sensitive-info-leak-servlet
 * @tags security
 *       external/cwe/cwe-536
 * @cwe CWE-536
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.DataFlow

module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {

  query predicate edges(PathNode a, PathNode b, string key, string val) {
    Merged::PathGraph::edges(a, b, key, val)
  }

  predicate isSource(DataFlow::Node source) { 
    exists(MethodAccess ma |
      // Sources from exceptions
      ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
      source.asExpr() = ma
    )
    or
    exists(MethodAccess ma |
      // Additional sources: Sensitive file paths
      ma.getMethod().hasName("getAbsolutePath") and
      ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
      source.asExpr() = ma
    )
   }
  predicate isSink(DataFlow::Node sink) { 
    exists(MethodAccess ma |
      // Sinks to the servlet response
      ma.getMethod().hasName("write") and
      ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
      ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
      sink.asExpr() = ma.getAnArgument()
    ) or
    exists(MethodAccess ma |
      // Sinks using PrintWriter
      ma.getMethod().hasName("println") and
      ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      sink.asExpr() = ma.getAnArgument()
    ) 
  }

}


module MyFlow = DataFlow::Global<SensitiveInfoLeakServletConfig>;

from MyFlow::PathNode source, MyFlow::PathNode sink
where MyFlow::flowPath(source, sink)
select sink, source, sink, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information."






// class SensitiveInfoLeakServletConfig extends TaintTracking::Configuration {
//   SensitiveInfoLeakServletConfig() { this = "SensitiveInfoLeakServletConfig" }


//   override predicate isSource(DataFlow::Node source) {
//     exists(MethodAccess ma |
//       // Sources from exceptions
//       ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
//       (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
//       source.asExpr() = ma
//     )
//     or
//     exists(MethodAccess ma |
//       // Additional sources: Sensitive file paths
//       ma.getMethod().hasName("getAbsolutePath") and
//       ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
//       source.asExpr() = ma
//     )
//   }

//   override predicate isSink(DataFlow::Node sink) {
//     exists(MethodAccess ma |
//       // Sinks to the servlet response
//       ma.getMethod().hasName("write") and
//       ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
//       ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
//       sink.asExpr() = ma.getAnArgument()
//     ) or
//     exists(MethodAccess ma |
//       // Sinks using PrintWriter
//       ma.getMethod().hasName("println") and
//       ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
//       sink.asExpr() = ma.getAnArgument()
//     )
//   }
// }

// from SensitiveInfoLeakServletConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
// where config.hasFlowPath(source, sink)
// select sink, source, sink, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information."





// private class GetMessageFlowSource extends DataFlow::Node {
//   GetMessageFlowSource() {
//     exists(MethodAccess ma |
//       // Sources from exceptions
//       ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
//       (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) 
//       // and
//       // source.asExpr() = ma
//     )
//     or
//     exists(MethodAccess ma |
//       // Additional sources: Sensitive file paths
//       ma.getMethod().hasName("getAbsolutePath") and
//       ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") 
//       // and
//       // source.asExpr() = ma
//     )
//   }
// }

// private class GetMessageFlowSink extends DataFlow::Node {
//   GetMessageFlowSink() {
//     exists(MethodAccess ma |
//       // Sinks to the servlet response
//       ma.getMethod().hasName("write") and
//       ma.getQualifier().(MethodAccess).getMethod().hasName("getWriter") and
//       ma.getQualifier().(MethodAccess).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") 
//       // and
//       // sink.asExpr() = ma.getAnArgument()
//     ) or
//     exists(MethodAccess ma |
//       // Sinks using PrintWriter
//       ma.getMethod().hasName("println") and
//       ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") 
//       // and
//       // sink.asExpr() = ma.getAnArgument()
//     )
//   }
// }
 
//  module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {
//   predicate isSource(DataFlow::Node source){ source instanceof GetMessageFlowSource }

//   predicate isSink(DataFlow::Node sink) { sink instanceof GetMessageFlowSink }
// }

//  module SensitiveInfoLeakServlet = TaintTracking::Global<SensitiveInfoLeakServletConfig>;
 
//  from SensitiveInfoLeakServlet::PathNode source, SensitiveInfoLeakServlet::PathNode sink
//  where SensitiveInfoLeakServlet::flowPath(source, sink)
//  select sink.getNode(), "Sensitive information might be exposed here."
