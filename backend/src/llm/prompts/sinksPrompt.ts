export const sinksPrompt = `
A sink is a point in the code where data exits a system. This is a critical point in the code where data can be exposed to an attacker.

## Sink Types
For this task, you have 13 different types of sinks to look for:
1) I/O Sink: Writes data to a file.
    Examples: FileWriter, FileOutputStream, FileChannel.
2) Print Sink: Prints data to the console.
    Examples: System.out.println, PrintWriter.
3) Network Sink: Sends data over the network.
    Examples: Socket, URL, HTTP connection, ServletResponse.
4) Log Sink: Logs data.
    Examples: Logger, System.out.println.
5) Database Sink: Writes data to a database.
    Examples: JDBC, JPA, Hibernate.
6) Email Sink: Sends data via email.
    Examples: JavaMail API, SMTP connections.
7) IPC Sink: Sends data between processes.
    Examples: Shared memory, named pipes, message queues.
8) Clipboard Sink: Writes data to the clipboard.
    Examples: java.awt.datatransfer.Clipboard.
9) GUI Display Sink: Displays data on a graphical user interface.
    Examples: JLabel, JTextField, JTextArea.
10) RPC Sink: Sends data via RPC mechanisms.
    Examples: RMI, gRPC.
11) Environment Variable Sink: Writes data to environment variables.
    Examples: System.setProperty.
12) Command Execution Sink: Executes commands potentially exposing data.
    Examples: Runtime.exec, ProcessBuilder.
13) Configuration File Sink: Writes data to configuration files.
    Examples: Writing to .properties, .xml, .json files.

### Example
If we have a sink such as system.out.println("Hello World"), the name of the sink would be "println". 
I don't need the system.out part, just the method name that is being called. 
Please apply this same logic to all the sinks that you find.
  
### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) If you find a sink I would like the name of the sink itself, along with a description of why you think it is a sink, and what type of sink it is.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the name of the sink. Just how it is given.
  B) The "isSink" field should be "yes" if you think this is a sink, and "no" if you don't think it is a sink.
  C) The "type" field should be the type of sink. Such as "I/O Sink" or "Print Sink".
  D) The "reason" field should be the reason why you think this is a sink or not.
{
  "files": [
    {
      "fileName": "FileName1.java",
      "sinks": [
        {
          "name": "string",
          "isSink": "yes/no",
          "type": "sinkType"
          "reason": "reason for the decision"
        }
      ]
    }
  ]
}`