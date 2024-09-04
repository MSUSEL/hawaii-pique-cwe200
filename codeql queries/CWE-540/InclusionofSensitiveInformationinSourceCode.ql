/**
 * @name CWE-540: Inclusion of Sensitive Information in Source Code (CWE-540)
 * @description Identifies instances where sensitive information such as credentials may be hardcoded in the source code, potentially leading to security vulnerabilities.
 * @kind problem
 * @problem.severity error
 * @id java/sensitive-info-in-code/540
 * @tags security
 *       external/cwe/cwe-540
 * @cwe CWE-540
 */

import java
import SensitiveInfo.SensitiveInfo

/**
 * Matches methods or constructors that:
 * - Have test annotations (@Test for JUnit 4 and JUnit 5)
 * - OR have the word 'test' in their names.
 */
class TestClassOrMethod extends Callable {
  TestClassOrMethod() {
    // JUnit 4 and JUnit 5 tests
    this.getAnAnnotation().getType().hasQualifiedName("org.junit", "Test") or
    this.getAnAnnotation().getType().hasQualifiedName("org.junit.jupiter.api", "Test") or
    this.getAnAnnotation()
        .getType()
        .hasQualifiedName("org.junit.jupiter.params", "ParameterizedTest") or
    // Methods with 'test' in their names
    this.getName().toLowerCase().regexpMatch(".*test.*") or
    // Class names with 'test' in their names
    this.getDeclaringType().getName().toLowerCase().regexpMatch(".*test.*")
  }
}

from SensitiveStringLiteral ssl
where
  // Exclude methods or classes that are tests
  not exists(TestClassOrMethod tcm | ssl.getEnclosingCallable() = tcm)
select ssl, "CWE-540 violation: sensitive information in source code."
