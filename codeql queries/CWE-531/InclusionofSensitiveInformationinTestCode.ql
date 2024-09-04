/**
 * @name CWE-531: Inclusion of Sensitive Information in Test Code
 * @description Detects potential information exposure through string literals in Java test code.
 * @kind problem
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
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo
private import semmle.code.java.dataflow.ExternalFlow

// module Flow = TaintTracking::Global<InclusionofSensitiveInformationinTestCodeConfig>;

// import Flow::PathGraph

// module InclusionofSensitiveInformationinTestCodeConfig implements DataFlow::ConfigSig {
//   predicate isSource(DataFlow::Node source) {
//     exists(SensitiveStringLiteral ssl, TestClass tc |
//       ssl.getEnclosingCallable().getDeclaringType() = tc and
//       source.asExpr() = ssl
//     )
//   }

//   predicate isSink(DataFlow::Node sink) {
//     // Ensure that all sinks are within servlets
//     exists(MethodCall mc |
//       sink.asExpr() = mc.getAnArgument() and
//       CommonSinks::isServletSink(sink)
//       or
//       CommonSinks::isPrintSink(sink)
//       or
//       CommonSinks::isErrorSink(sink)
//       or
//       //  Use the LLM response to indentify sinks
//       getSinkAny(sink)
//     )
//   }

//   predicate isBarrier(DataFlow::Node node) {
//     exists(MethodCall mc |
//       // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
//       (
//         mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
//         mc.getMethod().getName().toLowerCase().matches("%encrypt%")
//       ) and
//       // Consider both arguments and the return of sanitization/encryption methods as barriers
//       (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
//     )
//   }
// }

// class TestClass extends RefType {
//   TestClass() {
//     // JUnit 3 test case
//     this.getASupertype*().hasQualifiedName("junit.framework", "TestCase")
//     or
//     // JUnit 4 and 5 test annotations
//     this.getAnAnnotation()
//         .getType()
//         .hasQualifiedName(["org.junit", "org.junit.jupiter.api"], "Test")
//     or
//     // Test class naming convention
//     this.getName().regexpMatch(".*Test.*")
//   }
// }

// from Flow::PathNode source, Flow::PathNode sink
// where Flow::flowPath(source, sink)
// select sink.getNode(), source, sink, "CWE-531: Exposure of sensitive information in test code."


class TestClass extends RefType {
  TestClass() {
    // JUnit 3 test case
    this.getASupertype*().hasQualifiedName("junit.framework", "TestCase")
    or
    // JUnit 4 and 5 test annotations
    this.getAnAnnotation()
        .getType()
        .hasQualifiedName(["org.junit", "org.junit.jupiter.api"], "Test")
    or
    // Test class naming convention
    this.getName().regexpMatch(".*Test.*")
  }
}
// We want to find sensitive strings in code, test code
from SensitiveStringLiteral ssl,  TestClass tc
where ssl.getEnclosingCallable().getDeclaringType() = tc
select ssl, "CWE-531: Exposure of sensitive information in test code."
