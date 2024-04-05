/**
 * @name CWE-531: Information Exposure Through Variable in Java Test Code
 * @description Detects potential information exposure through string literals in Java test code.
 * @kind problem
 * @problem.severity medium
 * @precision medium
 * @id CWE-531
 * @tags security
 *       cwe-531
 *       java
 * @cwe CWE-531
 */
import java

import semmle.code.java.security.SensitiveVariables


 // Utilize the existing definitions for JUnit test classes
 class JUnit3TestClass extends Class {
     JUnit3TestClass() {
         this.getASupertype*().hasQualifiedName("junit.framework", "TestCase")
     }
 }
 
 class JUnit4Or5TestClass extends Class {
     JUnit4Or5TestClass() {
         this.getAnAnnotation().getType().hasQualifiedName(["org.junit", "org.junit.jupiter.api"], "Test")
     }
 }
 
 from Class testClass, Method m, SensitiveStringLiteral ssl
 where
     // Look for classes that are likely test classes
     (
     testClass instanceof JUnit3TestClass or
     testClass instanceof JUnit4Or5TestClass or
     testClass.getName().regexpMatch(".*Test.*") // Classes with 'Test' in their name
     ) and 
     // Get the methods within each test class
     m.getDeclaringType() = testClass and 
     // Find sensitive variable expressions within those methods
     ssl.getEnclosingCallable() = m
 
 select ssl, "CWE-531: Exposure of sensitive information in test code through a sensitive variable."
 