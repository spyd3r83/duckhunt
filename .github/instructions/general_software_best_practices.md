
# 🧭 General-Purpose Software Engineering: Best Practices Guide

## Summary

This guide provides universal, framework-agnostic best practices to improve **clarity**, **maintainability**, **security**, and **scalability** across any software project. It applies to any language, stack, or domain and is suitable for both humans and LLMs acting in engineering contexts.

> 🔁 **Always reason first, then act.** Show your reasoning steps *before* giving outputs, code, or conclusions.

**DO NOT BLOAT THE PROJECT WITH EXTRA FILES

---

## 📐 Code Style & Standards

- **Use enums instead of hardcoded strings**  
  *Avoids typos, eases refactoring, provides structure.*

    ```ts
    enum UserRole {
      Admin,
      Editor,
      Viewer
    }

    function canEdit(role: UserRole) {
      return role === UserRole.Admin || role === UserRole.Editor
    }
    ```

- **Prefer strong typing and modern language features**  
  *Improves IDE support, prevents runtime errors.*

    ```ts
    function sendMessage(user: { id: string, email: string }): boolean {
      // [...]
      return true
    }
    ```

- **Group related logic; use single-responsibility principle**  
  *Easier testing, change, and reuse.*

- **Apply consistent naming conventions**  
  *e.g., camelCase for variables, PascalCase for types/classes.*

- **Avoid deep nesting and long functions**  
  *Use early returns and helper methods to simplify logic.*

---

## 🌐 API & Service Design

- **Use versioned and RESTful or RPC interfaces with clear contracts**

    ```http
    POST /v1/users/
    GET /v1/users/{id}
    ```

- **Request/response schema must be formalized (e.g., OpenAPI, JSON Schema)**

    ```json
    {
      "id": "string",
      "email": "string",
      "roles": ["Admin", "Viewer"]
    }
    ```

- **Include pagination, sorting, filtering, and rate-limiting standards**

- **Avoid leaking implementation details in error responses**

---

## ❌ Error Handling

- **Prefer fail-fast and descriptive error messages**

    ```pseudo
    if (user.id is null) {
      throw new Error("Missing user ID")
    }
    ```

- **Standardize error codes and formats across the system**

    ```json
    {
      "error_code": "INVALID_INPUT",
      "message": "Email format is invalid",
      "trace_id": "req-12345"
    }
    ```

- **Always wrap external failures with contextual information**
- **Support graceful degradation where possible (fallbacks, retries)**

---

## ⚙️ Asynchronous Operations

- **Use async patterns for all I/O-bound tasks**

    ```js
    const user = await db.findUserById(id)
    const emailSent = await sendEmail(user.email)
    ```

- **Manage task cancellation, retries, and backoffs explicitly**

    ```pseudo
    retry(fetchRemoteData, strategy=exponentialBackoff(1s, 3x))
    ```

- **Use queues or workers for long-running background operations**

---

## 🐳 Containerization & Deployment

- **Use minimal, pinned base images**
- **Do not run as root; define unprivileged users**
- **Inject secrets via environment variables or secret stores**

    ```dockerfile
    FROM alpine:3.19
    RUN adduser -D appuser
    USER appuser
    ```

- **Health checks, liveness probes, and readiness probes are mandatory**

- **Always set resource requests/limits (e.g., CPU/memory)**

---

## 🧪 Testing

- **Write unit, integration, and e2e tests for all features**
- **Follow arrange-act-assert (AAA) test structure**
- **Test both happy paths and failure modes**

    ```pseudo
    test "should reject expired token":
        given expired token
        when validateToken(token)
        then error "TokenExpired"
    ```

- **Mock external dependencies, but also run contract tests**

---

## ⚙️ Environment & Configuration

- **Use `.env` or secure vaults for sensitive data; never commit secrets**
- **Support multiple environments via env var injection or config files**

    ```env
    APP_ENV=production
    DATABASE_URL=postgres://user:pass@db:5432/app
    ```

- **Validate configuration at startup with fail-fast behavior**

---

## 🔌 External Service Integration

- **Always timeout and retry with circuit breakers or backoff**

    ```pseudo
    if (response.time > 10s) then fail("Timeout contacting payment provider")
    ```

- **Log all interactions with unique identifiers**

- **Do not expose upstream error messages to clients**

- **Securely sign or encrypt requests when needed (HMAC, JWT, etc.)**

---

## 🧾 Logging & Monitoring

- **Use structured logs (JSON or key-value pairs)**  
  *Easy parsing, searchable, and alertable.*

    ```json
    {
      "level": "error",
      "message": "User creation failed",
      "context": { "email": "foo@example.com" }
    }
    ```

- **Include unique trace or request IDs in all logs**

- **Monitor system metrics (latency, error rates, uptime, etc.)**

- **Expose `/health`, `/metrics`, and `/ready` endpoints**

---

## 🔐 Security

- **Sanitize all input; validate types, formats, and ranges**

    ```pseudo
    if (!isValidEmail(input.email)) throw "BadRequest"
    ```

- **Use HTTPS, secure headers, and up-to-date dependencies**

- **Follow principle of least privilege in all roles and access**

- **Secure credentials, keys, and tokens with vaults and access policies**

- **Audit sensitive operations and login attempts**

---

## ⚡ Performance & Scalability

- **Favor stateless components**
- **Reuse connections (e.g., DB, HTTP clients)**

- **Use batching, caching, or CDN where appropriate**

    ```pseudo
    cache.set("user:123", userProfile, ttl=3600)
    ```

- **Avoid N+1 queries; profile and monitor critical paths**

---

## 📖 Documentation

- **Auto-generate API and schema documentation**
- **README.md must include setup, run, test, deploy instructions**

    ```markdown
    ## Run locally

    ```bash
    docker-compose up --build
    ```

    ## Test

    ```bash
    npm test
    ```
    ```

- **Document assumptions, edge cases, and limitations clearly**

- **Include visual diagrams where applicable (e.g., system or flowcharts)**

---

## 🌳 Version Control

- **Use semantic commit messages and conventional commits**

    ```
    feat(api): add endpoint to create orders
    fix(auth): handle expired refresh tokens
    ```

- **Feature branches must be merged via PRs with review**

- **Do not commit build artifacts or secrets**

---

## 🔁 Change Management

- **Understand dependencies before making changes**
- **Update changelogs and version numbers for each release**
- **Backwards compatibility must be preserved unless documented**

---

## 🧩 Patterns & Architecture

- **Use inversion of control or dependency injection where applicable**
- **Apply layered architecture (e.g., controller/service/repo layers)**
- **Use message queues and event-based systems for scalability**
- **Isolate side effects (e.g., DB writes, HTTP calls) in testable units**

---

## 🧠 Edge Cases & Special Considerations

- **Timeouts and Retries**: Every remote call must handle both.
- **Missing Configuration**: Fail early with explicit messages.
- **Race Conditions**: Locking or idempotency checks must be used.
- **Partial Failures**: Return degraded results where possible, not total failure.
- **Platform Differences**: Avoid system-dependent logic when cross-platform support is needed.

### Idempotent Bash Scripting  
- **Use `set -euo pipefail`** to ensure safe execution and fail fast.  
- **Design scripts to be safe for repeated runs without causing duplicate side effects.**  
- **Always check state before acting:**  
  - Verify if a file exists before creating or deleting.  
  - Confirm a package is already installed before attempting installation.  
  - Check if a user or group exists before adding.  
- **Guard changes with conditional logic (`if`, `grep`, `id`, `test -f`).**  
- **Favor declarative approaches that express the desired end state over imperative step-by-step changes.**  
- **Document *why* each operation is idempotent for clarity and maintainability.**  


---

## ✅ Final Reminder

To build reliable, secure, and maintainable software:

- **Reason before results**: Always explain *why* before *what*.
- **Use enums instead of strings**
- **Validate everything, fail fast, and log meaningfully**
- **Test, document, and version everything**
- **Keep it modular, stateless, and secure**

> ✨ *Write code your future self and teammates will thank you for.*
