/**
 * @name Information exposure through transmitted data
 * @description Transmitting sensitive information to the user is a potential security risk.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 4.3
 * @precision high
 * @id java/sensitive-data-transmission
 * @tags security
 *       external/cwe/cwe-201
 */

 import java
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.dataflow.flowsinks.Remote
 import semmle.code.java.frameworks.JDK
 import ExposureInTransmittedData::PathGraph
 
 module ExposureInTransmittedDataConfig extends DataFlow::Configuration {
   ExposureInTransmittedDataConfig() { this = "ExposureInTransmittedDataConfig" }
 
   override predicate isSource(DataFlow::Node source) {
     // `source` may contain a password
     source.asExpr() instanceof PasswordExpr
     or
     // `source` is from a `SQLException` property
     exists(MethodAccess ma, Method m |
       source.asExpr() = ma and
       ma.getQualifier().getType() = any(JDK::SQLException se).getASubType*() and
       m = ma.getMethod()
     |
       m.getName() = "getMessage" or
       m.getName() = "getErrorCode"
     )
     or
     // `source` is from `Throwable.toString()`
     exists(MethodCall mc |
       source.asExpr() = mc and
       mc.getQualifier().getType().getASupertype*() instanceof JDK::Throwable and
       mc.getMethod() = JDK::Throwable.toString()
     )
   }
 
   override predicate isSink(DataFlow::Node sink) { sink instanceof RemoteFlowSink }
 }
 
 class ExposureInTransmittedData extends TaintTracking::Configuration {
   ExposureInTransmittedData() { this = "ExposureInTransmittedData" }
 
   override predicate isSource(DataFlow::Node source) { source instanceof ExposureInTransmittedDataConfig::Source }
 
   override predicate isSink(DataFlow::Node sink) { sink instanceof ExposureInTransmittedDataConfig::Sink }
 }
 
 from ExposureInTransmittedData::SourceNode source, ExposureInTransmittedData::SinkNode sink
 where hasFlow(source, sink)
 select sink.getNode(), source, sink, "This data transmitted to the user depends on $@.",
   source.getNode(), "sensitive information"
 