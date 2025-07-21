/**
 * @name Local information disclosure in a temporary directory with sensitive data
 * @kind path-problem
 * @id java/local-temp-file-sensitive-data-disclosure/538
 * @problem.severity error
 * @precision high
 * @tags security
 *      external/cwe/cwe-538
 * @cwe CWE-538
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.frameworks.Servlets
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.DataFlow
 import Barrier.Barrier
 import SensitiveInfo.SensitiveInfo
 
 module Flow = TaintTracking::Global<SensitiveInfoLeakServletConfig>;
 
 module TempFileFlow = TaintTracking::Global<TempFileConfig>;
 
 module TempFileConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(MethodCall getPropertyCall, NewClassExpr fileCreation |
       getPropertyCall.getMethod().hasQualifiedName("java.lang", "System", "getProperty") and
       getPropertyCall.getArgument(0).(StringLiteral).getValue() = "java.io.tmpdir" and
       // Use DataFlow::localFlow instead of localExprFlow to handle variable flow
       DataFlow::localFlow(DataFlow::exprNode(getPropertyCall), DataFlow::exprNode(fileCreation.getArgument(0))) and
       fileCreation.getConstructedType().hasQualifiedName("java.io", "File") and
       source.asExpr() = fileCreation
     )
   }
 
   predicate isSink(DataFlow::Node sink) {
     // Sink is a File used in a FileWriter constructor
     exists(NewClassExpr writerCreation |
       writerCreation.getConstructedType().hasQualifiedName("java.io", "FileWriter") and
       sink.asExpr() = writerCreation.getAnArgument()
     )
   }
 }
 
 module SensitiveInfoLeakServletConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   predicate isSink(DataFlow::Node sink) {
     exists(MethodCall mc |
       mc.getMethod().hasName("write") and
       mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "Writer") and
       sink.asExpr() = mc.getAnArgument() and
       // Ensure the Writer is a FileWriter constructed from a temporary File
       exists(NewClassExpr writerCreation |
         writerCreation.getConstructedType().hasQualifiedName("java.io", "FileWriter") and
         DataFlow::localFlow(DataFlow::exprNode(writerCreation), DataFlow::exprNode(mc.getQualifier())) and
         TempFileFlow::flowToExpr(writerCreation.getAnArgument())
       )
     )
   }
 
   predicate isBarrier(DataFlow::Node node) {
     Barrier::barrier(node)
   }
 }
 
 import Flow::PathGraph
 
 predicate isTestFile(File f) {
   exists(string path | path = f.getAbsolutePath().toLowerCase() |
     path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
   )
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink) and
       not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink,
   "CWE-538: Insertion of Sensitive Information into Externally-Accessible File or Directory"