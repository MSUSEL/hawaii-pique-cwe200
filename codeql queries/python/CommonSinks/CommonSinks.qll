// This is a helper lib that contains a set of common sinks. This is not exhaustive and should be used with ChatGPT.

import python
import semmle.python.dataflow.new.DataFlow

module CommonSinks {

    predicate isLoggingSink(DataFlow::Node sink) {
        exists(Call call |
          // logging module calls
          call.getFunc().(Attribute).getName() in ["error", "warn", "warning", "info", "debug", "critical", "fatal"] and
          call.getFunc().(Attribute).getObject().(Name).getId() = "logging" and
          sink.asExpr() = call.getAnArg()
        )
        or
        exists(Call call |
          // logger instance calls
          call.getFunc().(Attribute).getName() in ["error", "warn", "warning", "info", "debug", "critical", "fatal"] and
          sink.asExpr() = call.getAnArg()
        )
    }

    predicate isWebFrameworkSink(DataFlow::Node sink) {
        exists(Call call |
            // Flask response methods
            call.getFunc().(Attribute).getName() in ["make_response", "jsonify", "abort"] and
            sink.asExpr() = call.getAnArg()
        )
        or
        exists(Call call |
            // Django HttpResponse
            call.getFunc().(Name).getId() in ["HttpResponse", "JsonResponse", "Http404"] and
            sink.asExpr() = call.getAnArg()
        )
        or
        exists(Call call |
            // FastAPI responses
            call.getFunc().(Attribute).getName() in ["JSONResponse", "PlainTextResponse", "HTMLResponse"] and
            sink.asExpr() = call.getAnArg()
        )
    }

    predicate isPrintSink(DataFlow::Node sink) {
        // print() function calls
        exists(Call call |
          call.getFunc().(Name).getId() = "print" and
          sink.asExpr() = call.getAnArg()
        )
        or
        // sys.stdout.write() and sys.stderr.write()
        exists(Call call |
          call.getFunc().(Attribute).getName() = "write" and
          call.getFunc().(Attribute).getObject().(Attribute).getName() in ["stdout", "stderr"] and
          call.getFunc().(Attribute).getObject().(Attribute).getObject().(Name).getId() = "sys" and
          sink.asExpr() = call.getAnArg()
        )
        or
        // File write operations
        exists(Call call |
          call.getFunc().(Attribute).getName() in ["write", "writelines"] and
          sink.asExpr() = call.getAnArg()
        )
    }

      
      

    predicate isErrPrintSink(DataFlow::Node sink) {
        exists(Call call |
            // sys.stderr.write() or print(..., file=sys.stderr)
            (
              call.getFunc().(Attribute).getName() = "write" and
              call.getFunc().(Attribute).getObject().(Attribute).getName() = "stderr" and
              call.getFunc().(Attribute).getObject().(Attribute).getObject().(Name).getId() = "sys"
            ) or (
              call.getFunc().(Name).getId() = "print" and
              exists(Keyword kw | 
                kw = call.getAKeyword() and 
                kw.getArg() = "file" and
                kw.getValue().(Attribute).getName() = "stderr"
              )
            ) and
            sink.asExpr() = call.getAnArg()
        )
    }
    

    predicate isErrorSink(DataFlow::Node sink) {
        exists(Call call |
            // traceback.print_exc(), traceback.format_exc()
            call.getFunc().(Attribute).getName() in ["print_exc", "format_exc", "print_tb", "format_tb"] and
            call.getFunc().(Attribute).getObject().(Name).getId() = "traceback" and
            sink.asExpr() = call
        )
        or
        exists(Attribute attr |
            // Exception.__traceback__, Exception.args
            attr.getName() in ["__traceback__", "args"] and
            sink.asExpr() = attr
        )
        or
        exists(Call call |
            // str(exception) to get string representation
            call.getFunc().(Name).getId() = "str" and
            sink.asExpr() = call.getAnArg()
        )
    }

    predicate isIOSink(DataFlow::Node sink) {
        // File writing operations
        exists(Call call |
            (
              call.getFunc().(Attribute).getName() in ["write", "writelines"] or
              call.getFunc().(Name).getId() in ["open"] // open() when used for writing
            ) and
            sink.asExpr() = call.getAnArg()
        )
        or
        // JSON/pickle serialization
        exists(Call call |
            (
              call.getFunc().(Attribute).getName() in ["dump", "dumps"] and
              call.getFunc().(Attribute).getObject().(Name).getId() in ["json", "pickle"]
            ) and
            sink.asExpr() = call.getAnArg()
        )
    }

    predicate isPythonFrameworkSink(DataFlow::Node sink) {
        exists(Call call |
            // Flask error handling
            call.getFunc().(Name).getId() in ["abort", "make_response", "jsonify"] and
            sink.asExpr() = call.getAnArg()
        )
        or
        exists(Call call |
            // Django responses
            call.getFunc().(Name).getId() in ["HttpResponse", "JsonResponse", "Http404", "HttpResponseServerError"] and
            sink.asExpr() = call.getAnArg()
        )
        or
        exists(Call call |
            // FastAPI/Starlette responses
            call.getFunc().(Attribute).getName() in ["JSONResponse", "PlainTextResponse", "HTMLResponse"] and
            sink.asExpr() = call.getAnArg()
        )
    }
}
