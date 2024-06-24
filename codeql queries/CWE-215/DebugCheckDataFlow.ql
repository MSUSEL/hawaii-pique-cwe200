/**
 * @name CWE-215: Sensitive data exposure in debug code
 * @description Identifies instances where sensitive variables are exposed in debug output without proper sanitization, which could lead to sensitive data leaks.
 * @kind path-problem
 * @problem.severity warning
 * @id java/debug-code-sensitive-data-exposure/215
 * @tags security
 *      external/cwe/cwe-215
 *      external/cwe/cwe-532
 * @cwe CWE-215
 */

import java
import SensitiveInfo.SensitiveInfo
module Flow = TaintTracking::Global<DebugCheckConfig>;
import Flow::PathGraph
import semmle.code.java.dataflow.DataFlow
import CommonSinks.CommonSinks

module DebugCheckConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    (
      exists(SensitiveVariableExpr sve | source.asExpr() = sve) or 
      exists(SensitiveStringLiteral ssl |  source.asExpr() = ssl) or
      exists(ExceptionOrMethodCall eomc| source.asExpr() = eomc) 
      ) 
      and  
    isDebugGuarded(source)
  }

  predicate isSink(DataFlow::Node sink) {
      getSinkAny(sink) or
      CommonSinks::isLoggingSink(sink) or
      CommonSinks::isPrintSink(sink) or
      CommonSinks::isServletSink(sink) or
      CommonSinks::isErrorSink(sink) or
      CommonSinks::isIOSink(sink)
      // Use the LLM response to indentify sinks
      or getSinkAny(sink)
  }

  predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%")) and
    // Consider both arguments and the return of sanitization/encryption methods as barriers
    (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
    )
  }
}

class DebugCondition extends IfStmt {
  DebugCondition() {
    exists(Expr condition |
      this.getCondition() = condition and
      condition.toString().toLowerCase().regexpMatch(".*debug.*")
    )
  }
}

predicate isDebugGuarded(DataFlow::Node node) {
  exists(DebugCondition debugCond |
    debugCond.getAChild*() = node.asExpr().getEnclosingStmt()
  )
}

class ExceptionType extends RefType {
  ExceptionType() {
    this.getASupertype*().hasQualifiedName("java.lang", "Throwable")
  }
}

class ExceptionOrMethodCall extends Expr {
  ExceptionOrMethodCall() {
    // Direct exception references (e)
    exists(VarAccess va |
      va.getType() instanceof ExceptionType and
      this = va
    )
    or
    // Calls to methods on exceptions, like getMessage() and toString()
    exists(MethodCall ma, ExceptionType et |
      ma.getQualifier().getType() = et and
      (
        ma.getMethod().getName() = "getMessage" or
        ma.getMethod().getName() = "toString"
      ) and
      this = ma
    )
  }
}


from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink) 
select sink.getNode(), source, sink,
  "CWE-215: Sensitive data exposure in debug code"
