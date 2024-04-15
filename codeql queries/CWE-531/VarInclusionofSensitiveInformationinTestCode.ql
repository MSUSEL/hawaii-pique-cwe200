/** 
 * @name CWE-531: Information Exposure Through Variable in Java Test Code
 * @description Detects potential information exposure through string literals in Java test code.
 * @kind path-problem
 * @problem.severity medium
 * @precision medium
 * @id java/test-code-sensitive-var-info-exposure/531
 * @tags security
 *       cwe-531
 *       java
 * @cwe CWE-531
 * @severity medium

 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.frameworks.Servlets
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.DataFlow
 import semmle.code.java.security.SensitiveVariables
 import CommonSinks.CommonSinks
 
 module Flow = TaintTracking::Global<InclusionofSensitiveInformationinTestCodeConfig>;
 import Flow::PathGraph
 
 module InclusionofSensitiveInformationinTestCodeConfig implements DataFlow::ConfigSig {
 
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve, TestClass tc |
        sve.getEnclosingCallable().getDeclaringType() = tc and
        source.asExpr() = sve
     )
   }
 
   predicate isSink(DataFlow::Node sink) {
     // Ensure that all sinks are within servlets
     exists(MethodCall mc | sink.asExpr() = mc.getAnArgument() and
     CommonSinks::isServletSink(sink) or
     CommonSinks::isPrintSink(sink) or
     CommonSinks::isLoggingSink(sink)) or 
     CommonSinks::isErrorSink(sink)
   }
 }

 class TestClass extends RefType {
  TestClass() {
      // JUnit 3 test case
      this.getASupertype*().hasQualifiedName("junit.framework", "TestCase") or
      // JUnit 4 and 5 test annotations
      this.getAnAnnotation().getType().hasQualifiedName(["org.junit", "org.junit.jupiter.api"], "Test") or
      // Test class naming convention
      this.getName().regexpMatch(".*Test.*")
  }
}
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "CWE-531: Exposure of sensitive information in test code."
 