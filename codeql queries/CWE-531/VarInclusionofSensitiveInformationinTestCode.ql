/** 
 * @name Exposure of sensitive information in servlet responses
 * @description Writing sensitive information from exceptions or sensitive file paths to HTTP responses can leak details to users.
 * @kind path-problem
 * @problem.severity warning
 * @id java/servlet-info-exposure/CWE-536
 * @tags security
 *       external/cwe/cwe-536
 * @cwe CWE-536
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
 
/**
 * The purpose of this query is to detect potential information exposure through sensitive variables in Java test code.
 */ 
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
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "CWE-531: Exposure of sensitive information in test code."
 