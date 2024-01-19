import java

class SensitiveInfoStringLiteral extends StringLiteral {
    SensitiveInfoStringLiteral() {
        this.getValue().toLowerCase().matches("%password%") or
        this.getValue().toLowerCase().matches("%username%") or
        this.getValue().toLowerCase().matches("%admin%") or
        this.getValue().toLowerCase().matches("%authentication%") or
        this.getValue().toLowerCase().matches("%authorization%") or
        this.getValue().toLowerCase().matches("%private%") or
        this.getValue().toLowerCase().matches("%sensitive%") or
        this.getValue().toLowerCase().matches("%secret%") or
        this.getValue().toLowerCase().matches("%token%") or
        this.getValue().toLowerCase().matches("%key%") or
        this.getValue().toLowerCase().matches("%credential%") or
        this.getValue().toLowerCase().matches("%temporary solution%") or
        this.getValue().toLowerCase().matches("%workaround%") or
        this.getValue().toLowerCase().matches("%for testing only%") or
        this.getValue().toLowerCase().matches("%remove before production%") or
        this.getValue().toLowerCase().matches("%debug code%") or
        this.getValue().toLowerCase().matches("%http://%") or
        this.getValue().toLowerCase().matches("%https://%") or
        this.getValue().toLowerCase().matches("%C:\\%") or
        this.getValue().toLowerCase().matches("%/etc/%") or
        this.getValue().toLowerCase().matches("%/home/%") or
        this.getValue().toLowerCase().matches("%/users/%") or
        this.getValue().toLowerCase().matches("%@deprecated%") or
        this.getValue().toLowerCase().matches("%reviewed by%")
    }
}

from SensitiveInfoStringLiteral lit
select lit, "This string literal might contain sensitive information."
