/**
 * @name CWE-598: Sensitive GET Query
 * @description Use of GET request method with sensitive query strings.
 * @kind path-problem
 * @problem.severity warning
 * @precision medium
 * @id java/sensitive-get-query/598
 * @tags security
 *       experimental
 *       external/cwe/cwe-598
 * @cwe CWE-598
 */

import java
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveActions
import SensitiveGetQueryFlow::PathGraph
import SensitiveInfo.SensitiveInfo

module SensitiveGetQueryFlow = TaintTracking::Global<SensitiveGetQueryConfig>;

/** A variable that holds sensitive information judging by its name. */
class SensitiveInfoExpr extends Expr {
  SensitiveInfoExpr() {
    exists(Variable v | this = v.getAnAccess() |
      v.getName().regexpMatch(getCommonSensitiveInfoRegex()) and
      not v.getName().matches("token%") // exclude ^token.* since sensitive tokens are usually in the form of accessToken, authToken, ...
      and not v.getName().matches("%encrypt%") 
    )
    or 
    exists(SensitiveVariableExpr sve |
      this = sve and
      not sve.toString().matches(".*encrypt.*")
    )
  }
}

/** Holds if `m` is a method of some override of `HttpServlet.doGet`. */
private predicate isGetServletMethod(Method m) {
  isServletRequestMethod(m) and m.getName() = "doGet"
}

/** The `doGet` method of `HttpServlet`. */
class DoGetServletMethod extends Method {
  DoGetServletMethod() { isGetServletMethod(this) }
}

/** Holds if `ma` is (perhaps indirectly) called from the `doGet` method of `HttpServlet`. */
predicate isReachableFromServletDoGet(Call ma) {
  ma.getEnclosingCallable() instanceof DoGetServletMethod
  or
  exists(Method pm, Call pma |
    ma.getEnclosingCallable() = pm and
    pma.getCallee() = pm and
    isReachableFromServletDoGet(pma)
  )
}

/** Source of GET servlet requests. */
class RequestGetParamSource extends DataFlow::ExprNode {
  RequestGetParamSource() {
    exists(Call ma |
      isRequestGetParamMethod(ma) and
      ma = this.asExpr() and
      isReachableFromServletDoGet(ma)
    )
  }
}

/** A taint configuration tracking flow from the `ServletRequest` of a GET request handler to an expression whose name suggests it holds security-sensitive data. */
module SensitiveGetQueryConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { source instanceof RequestGetParamSource }

  predicate isSink(DataFlow::Node sink) { sink.asExpr() instanceof SensitiveInfoExpr or sink.asExpr() instanceof SensitiveVarAccess }

  /** Holds if the node is in a servlet method other than `doGet`. */
  predicate isBarrier(DataFlow::Node node) {
    isServletRequestMethod(node.getEnclosingCallable()) and
    not isGetServletMethod(node.getEnclosingCallable())
  }
}


from SensitiveGetQueryFlow::PathNode source, SensitiveGetQueryFlow::PathNode sink
where SensitiveGetQueryFlow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "$@ uses the GET request method to transmit sensitive information.", source.getNode(),
  "This request"
