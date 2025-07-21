/**
 * @name CWE-201: Insertion of Sensitive Information Into Sent Data
 * @description Information that is sensitive is sent to an external entity.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 4.3
 * @precision high
 * @id python/transmitted-data-exposure/201
 * @tags security
 *       external/cwe/cwe-201
 * @cwe CWE-201
 */

 import python
 import semmle.python.dataflow.new.DataFlow
 import semmle.python.dataflow.new.TaintTracking
 import SensitiveInfo.SensitiveInfo
 
 module ExposureInTransmittedData = TaintTracking::Global<ExposureInTransmittedDataConfig>;
 import ExposureInTransmittedData::PathGraph

 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){source.asExpr() instanceof SensitiveVariableExpr}

  predicate isSink(DataFlow::Node sink) {
    // Custom sinks from SensitiveInfo
    // getSink(sink, "Network Sink") or
    // getSink(sink, "Email Sink") or
    // getSink(sink, "I/O Sink") or
      
// 1) HTTP response headers (using requests, flask, django, etc.)
exists(Call call |
  (
    // Flask response headers
    call.getFunc().(Attribute).getName() in ["add", "set"] and
    call.getFunc().(Attribute).getObject().(Attribute).getName() = "headers" or
    // Django HttpResponse
    call.getFunc().(Attribute).getName() = "setdefault" and
    call.getFunc().(Attribute).getObject().(Name).getId() = "META" or
    // requests headers
    call.getFunc().(Name).getId() in ["post", "get", "put", "patch", "delete"] and
    exists(Keyword kw | kw = call.getAKeyword() and kw.getArg() = "headers")
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 2) Email sending (smtplib, email package)
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["send_message", "sendmail", "send"] or
    call.getFunc().(Attribute).getName() = "set_content" or
    call.getFunc().(Name).getId() in ["send_message", "sendmail"]
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 3) File writing operations
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["write", "writelines"] or
    call.getFunc().(Name).getId() in ["write", "open"]
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 4) Socket operations
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["send", "sendall", "sendto"] or
    call.getFunc().(Name).getId() in ["send", "sendall", "sendto"]
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 5) HTTP client libraries (requests, urllib, httpx)
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["post", "put", "patch", "request"] or
    call.getFunc().(Name).getId() in ["post", "put", "patch", "request", "urlopen"]
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 6) Web framework response writing (Flask, Django, FastAPI)
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["write", "flush", "make_response"] or
    call.getFunc().(Name).getId() in ["make_response", "jsonify"]
  ) and
  sink.asExpr() = call.getAnArg()
)

or

// 7) JSON/XML serialization that might be transmitted
exists(Call call |
  (
    call.getFunc().(Attribute).getName() in ["dumps", "dump"] or
    call.getFunc().(Name).getId() in ["dumps", "dump", "tostring"]
  ) and
  sink.asExpr() = call.getAnArg()
)
 }

  predicate isBarrier(DataFlow::Node node) {
    // Add any Python-specific barriers here if needed
    none()
  }
}


predicate isTestFile(File f) {
  // Convert path to lowercase for case-insensitive matching
  exists(string path | path = f.getAbsolutePath().toLowerCase() |
    // Check for common test-related directory or file name patterns
    path.regexpMatch(".*(test|tests|testing|test_suite|testcase|unittest|integration_test|spec).*")
  )
}

 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
 and not isTestFile(sink.getNode().getLocation().getFile())

  select sink.getNode(), source, sink,
  "CWE-201: Insertion of Sensitive Information Into Sent Data"
 

  