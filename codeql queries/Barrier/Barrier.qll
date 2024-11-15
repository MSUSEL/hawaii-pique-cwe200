import java
import semmle.code.java.dataflow.DataFlow

module Barrier {
    predicate barrier(DataFlow::Node node) {
        exists(MethodCall mc |
            // Detect common sanitization, encryption, encoding, and masking terms
            (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
             mc.getMethod().getName().toLowerCase().matches("%encrypt%") or
             mc.getMethod().getName().toLowerCase().matches("%encode%") or
             mc.getMethod().getName().toLowerCase().matches("%mask%") or
             mc.getMethod().getName().toLowerCase().matches("%hash%") or
             mc.getMethod().getName().toLowerCase().matches("%obfuscate%") or
             mc.getMethod().getName().toLowerCase().matches("%scramble%") or
             mc.getMethod().getName().toLowerCase().matches("%anonymize%") or
             mc.getMethod().getName().toLowerCase().matches("%redact%")) and

            // Include arguments and the return value as barriers
            (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
        )
        or
        // Commonly used encoding and cryptographic classes
        exists(Expr e |
            e = node.asExpr() and
            (
                e.getType() instanceof RefType and
                (
                    e.getType().(RefType).hasQualifiedName("java.net", "URLEncoder") or
                    e.getType().(RefType).hasQualifiedName("java.util", "Base64") or
                    e.getType().(RefType).hasQualifiedName("javax.crypto", "Cipher") or
                    e.getType().(RefType).hasQualifiedName("java.security", "MessageDigest") or
                    e.getType().(RefType).hasQualifiedName("java.security", "Signature") or
                    e.getType().(RefType).hasQualifiedName("java.security", "KeyStore")
                )
            )
        )
        or
        // Specific methods from known security libraries, frameworks, or encoding utilities
        exists(MethodCall frameworkSanitize |
            (
                // OWASP ESAPI
                (
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("org.owasp.esapi", "ESAPI") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("org.apache.commons.codec.binary", "Base64") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("org.apache.commons.codec", "StringEncoder") or
                    // Google Guava for encoding and hashing
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("com.google.common.hash", "Hashing") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("com.google.common.io", "BaseEncoding") or
                    // Spring Security
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("org.springframework.security.crypto", "Encryptors") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("org.springframework.security.crypto.bcrypt", "BCryptPasswordEncoder") or
                    // Java JWT (JSON Web Token) encoding
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("io.jsonwebtoken", "JwtBuilder") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("io.jsonwebtoken", "JwtParser") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("io.jsonwebtoken", "Jwt") or
                    // Encoding in standard Java libraries
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("java.nio.charset", "CharsetEncoder") or
                    frameworkSanitize.getMethod().getDeclaringType().(RefType).hasQualifiedName("java.util.zip", "Deflater")
                )
            ) and
            // Apply to both arguments and return values
            (node.asExpr() = frameworkSanitize.getAnArgument() or node.asExpr() = frameworkSanitize)
        )
        or
        // Hashing or digest creation methods as barriers for sensitive information
        exists(MethodCall hashMethod |
            hashMethod.getMethod().getName().toLowerCase().matches("%hash%") and
            hashMethod.getMethod().getDeclaringType().(RefType).hasQualifiedName("java.security", "MessageDigest") and
            (node.asExpr() = hashMethod.getAnArgument() or node.asExpr() = hashMethod)
        )
        or
        // Advanced barriers for data anonymization, masking, or scrambling using custom classes
        exists(MethodCall customBarriers |
            (customBarriers.getMethod().getName().toLowerCase().matches("%anonymize%") or
             customBarriers.getMethod().getName().toLowerCase().matches("%mask%") or
             customBarriers.getMethod().getName().toLowerCase().matches("%scramble%")) and
            (node.asExpr() = customBarriers.getAnArgument() or node.asExpr() = customBarriers)
        )
    }
}
