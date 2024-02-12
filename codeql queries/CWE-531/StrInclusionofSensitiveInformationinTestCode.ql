/**
 * @name CWE-531: Information Exposure Through Variable in Java Test Code
 * @description Detects potential information exposure through string literals in Java test code.
 * @kind problem
 * @problem.severity medium
 * @precision medium
 * @id java/information-exposure-test-code
 * @tags security
 *       cwe-531
 *       java
 */
import java

class JUnit3TestClass extends Class {
    // A class that extends junit.framework.TestCase is considered a JUnit 3 test class
    JUnit3TestClass() {
        this.getASupertype*().hasQualifiedName("junit.framework", "TestCase")
    }
}

class JUnit4Or5TestClass extends Class {
    // A class containing methods annotated with JUnit 4 or JUnit 5 @Test annotations
    JUnit4Or5TestClass() {
        this.getAnAnnotation().getType().hasQualifiedName(["org.junit", "org.junit.jupiter.api"], "Test")
    }
}

from Class testClass, Method m, StringLiteral sl
where
    // Look for classes that are likely test classes
    (
    testClass instanceof JUnit3TestClass or
    testClass instanceof JUnit4Or5TestClass
    ) and 
    
    // Get the methods within each class
    m.getDeclaringType() = testClass and 
    (
        (
            sl.getEnclosingCallable() = m and 
            sl.getValue().toLowerCase().matches(["%username%", "%password%", "%secret%", "%key%", "%token%", "%api%"]) and
            // Exclude literals used from config files   
            not sl.getValue().regexpMatch([".*\\.username.*", ".*\\.password.*", ".*\\.secret.*", ".*\\.key.*", ".*\\.token.*", ".*\\.api.*"])
        ) 
    ) 

select sl, "Potential CWE-531: Variable Inclusion of Sensitive Information in Test Code via string literal"



