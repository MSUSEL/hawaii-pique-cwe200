/**
 * @name CWE-535: Information Exposure Through Shell Error Message from Sensitive Input
 * @description Detects flows where sensitive data influences shell commands and their error messages are exposed.
 * @kind path-problem
 * @problem.severity warning
 * @id java/shell-error-exposure-sensitive/535
 * @tags security
 *       external/cwe/cwe-535
 * @cwe CWE-535
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import SensitiveInfo.SensitiveInfo
import CommonSinks.CommonSinks
import Barrier.Barrier

/**
 * Define flow states for the three steps.
 */
private newtype ShellErrorFlowState =
  State1() or  // Sensitive data is introduced
  State2() or  // Sensitive data flows into a shell command execution
  State3()     // Error output from the shell command is exposed

/**
 * Configuration for tracking sensitive data through shell commands to error exposure.
 */
module ShellErrorExposureConfig implements DataFlow::StateConfigSig {
  class FlowState = ShellErrorFlowState;

  /**
   * Step 1: Sensitive data as the source (State1).
   */
  predicate isSource(DataFlow::Node source, FlowState state) {
    state = State1() and
    exists(SensitiveVariableExpr sve | source.asExpr() = sve)
  }

  /**
   * Step 3: Exposure points as sinks (State3).
   */
  predicate isSink(DataFlow::Node sink, FlowState state) {
    state = State3() and
    (CommonSinks::isErrPrintSink(sink) or
     CommonSinks::isErrorSink(sink) or
     CommonSinks::isServletSink(sink))
  }

  /**
   * Define transitions between states.
   */
  predicate isAdditionalFlowStep(
    DataFlow::Node node1, FlowState state1,
    DataFlow::Node node2, FlowState state2
  ) {
    // State1 -> State2: Sensitive data flows into a shell command execution
    state1 = State1() and
    state2 = State2() and
    (exists(MethodCall execCall |
      execCall.getMethod().getName() = "exec" and
      execCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
      node1.asExpr() = execCall.getAnArgument() and
      node2.asExpr() = execCall
    )
    or
    exists(ConstructorCall pbCall |
      pbCall.getConstructedType().hasQualifiedName("java.lang", "ProcessBuilder") and
      node1.asExpr() = pbCall.getAnArgument() and
      node2.asExpr() = pbCall
    ))

    or

    // State2 -> State3: Shell command Process flows to error stream
    state1 = State2() and
    state2 = State3() and
    exists(MethodCall getErrorStreamCall |
      getErrorStreamCall.getMethod().getName() = "getErrorStream" and
      getErrorStreamCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Process") and
      node1.asExpr() = getErrorStreamCall.getQualifier() and
      node2.asExpr() = getErrorStreamCall
    )

    or

    // State3: Propagate taint through stream handling to exposure
    state1 = State3() and
    state2 = State3() and
    (
      // InputStream -> InputStreamReader
      exists(ConstructorCall readerCall |
        readerCall.getConstructedType().hasQualifiedName("java.io", "InputStreamReader") and
        node1.asExpr() = readerCall.getAnArgument() and
        node2.asExpr() = readerCall
      )
      or
      // InputStreamReader -> BufferedReader
      exists(ConstructorCall bufferedReaderCall |
        bufferedReaderCall.getConstructedType().hasQualifiedName("java.io", "BufferedReader") and
        node1.asExpr() = bufferedReaderCall.getAnArgument() and
        node2.asExpr() = bufferedReaderCall
      )
      or
      // BufferedReader -> readLine result
      exists(MethodCall readLineCall |
        readLineCall.getMethod().hasName("readLine") and
        readLineCall.getMethod().getDeclaringType().hasQualifiedName("java.io", "BufferedReader") and
        node1.asExpr() = readLineCall.getQualifier() and
        node2.asExpr() = readLineCall
      )
      or
      // String concatenation with tainted data
      exists(AddExpr add |
        node2.asExpr() = add and
        (node1.asExpr() = add.getLeftOperand() or node1.asExpr() = add.getRightOperand())
      )
    )
  }

  predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
  }
}

module ShellErrorFlow = TaintTracking::GlobalWithState<ShellErrorExposureConfig>;
import ShellErrorFlow::PathGraph

from ShellErrorFlow::PathNode source, ShellErrorFlow::PathNode sink
where ShellErrorFlow::flowPath(source, sink)
select sink, source, sink, "CWE-535: Sensitive data flows into a shell command whose error message is exposed."