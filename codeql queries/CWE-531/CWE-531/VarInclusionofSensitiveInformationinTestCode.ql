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


from Class testClass, Method m, LocalVariableDeclStmt localVarDecl
where
    // Look for classes that are likely test classes
    (
    testClass instanceof JUnit3TestClass or
    testClass instanceof JUnit4Or5TestClass
    ) and 
    
    // Get the methods within each class
    m.getDeclaringType() = testClass and 
    (
        // Find local variable declarations within these methods
        localVarDecl = any(LocalVariableDeclStmt lv | lv.getEnclosingCallable() = m) and
        localVarDecl.getAVariable().getName().toLowerCase().matches(["%username%", "%password%", "%secret%", "%key%", "%token%"])
        // Ensure the localVarDelc is assigned to a string literal
        and localVarDecl.getAVariable().getAChildExpr() instanceof StringLiteral
    ) 


select localVarDecl, "Potential CWE-531: Variable Inclusion of Sensitive Information in Test Code via variable assigned to string literal"



