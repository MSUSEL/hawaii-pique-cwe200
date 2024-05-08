/**
 * @name CWE-531: Inclusion of Sensitive Information in Test Code
 * @description Detects potential information exposure through string literals in Java test code.
 * @kind problem
 * @problem.severity medium
 * @precision medium
 * @id java/test-code-sensitive-str-info-exposure/531
 * @tags security
 *       cwe-531
 *       java
 * @cwe CWE-531
 * @severity medium
 */
import java
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo

// The purpose of this query is to detect potential information exposure through sensitive strings and comments in Java test code.

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

from TestClass tc, Method m, StringLiteral sl
where
    m.getDeclaringType() = tc and 
    sl.getEnclosingCallable() = m and
    (
        // Check for sensitive string literals within the method
        exists(SensitiveStringLiteral ssl | ssl.getEnclosingCallable() = m) or
        // Check for sensitive comments within the method
        exists(SensitiveComment sc | sc.getEnclosingCallable() = m)
    )
select sl, "CWE-531: Exposure of sensitive information in test code within method " + m.getName() + "."
