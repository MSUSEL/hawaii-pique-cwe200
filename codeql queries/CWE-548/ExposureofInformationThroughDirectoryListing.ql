import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking

class ListFilesToPrintConfiguration extends TaintTracking::Configuration {
    ListFilesToPrintConfiguration() { this = "ListFilesToPrintConfig" }

    override predicate isSource(DataFlow::Node source) {
        exists(MethodCall ma |
            ma.getMethod().hasName("listFiles") and
            ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
            source.asExpr() = ma
        )
    }

    override predicate isSink(DataFlow::Node sink) {
        exists(MethodCall ma |
            ma.getMethod().hasName("println") and
            sink.asExpr() = ma.getAnArgument()
        )
    }
}

from ListFilesToPrintConfiguration config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink, source, "Potential CWE-548: Exposure of Information Through Directory Listing."
