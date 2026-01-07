---
applyTo: '**'
---

# Professional Human-Like Code Style Guide

## Core Principle

Write code and documentation as a professional software engineer would: clear, direct, and focused on technical precision. Avoid artificial flourishes, unnecessary decoration, or non-standard characters that distract from the technical content.

---

## Prohibited Practices

### No Emojis or Decorative Characters

**Never use emojis** in any of the following contexts:

- Source code files (comments, docstrings, string literals)
- Documentation (README, guides, technical specifications)
- Commit messages
- Code comments or inline documentation
- Variable names, function names, or identifiers
- Log messages or error output
- Test case descriptions
- Configuration files
- API responses or user-facing messages

**Examples of what NOT to do:**

```python
# ❌ BAD: Emojis in documentation
def process_data() -> None:
    """Process the data 🚀 and save results 💾."""
    logger.info("Starting processing... ⚡")
```

```powershell
# ❌ BAD: Decorative characters in comments
function Get-UserData {
    # 🎯 Main function to retrieve user information
    Write-Host "Success! ✅"
}
```

**Correct approach:**

```python
# ✓ GOOD: Clear, professional documentation
def process_data() -> None:
    """Process the data and persist results to storage."""
    logger.info("Starting data processing")
```

```powershell
# ✓ GOOD: Direct, technical language
function Get-UserData {
    # Retrieve user information from the directory service
    Write-Verbose "Successfully retrieved user data"
}
```

---

## Professional Communication Standards

### Documentation and Comments

Write as if you are a senior engineer communicating with peers:

- **Be direct**: State what the code does without embellishment
- **Be technical**: Use precise terminology appropriate to the domain
- **Be concise**: Avoid unnecessary words or explanations
- **Be factual**: Document behavior, not feelings or enthusiasm

**Examples:**

```python
# ❌ BAD: Over-enthusiastic, informal tone
def calculate_metrics() -> dict[str, float]:
    """
    This awesome function calculates all the super important metrics!
    It's really cool and will make your analysis so much better! 🎉
    """
    pass

# ✓ GOOD: Professional, technical tone
def calculate_metrics() -> dict[str, float]:
    """Calculate performance metrics from the current dataset.
    
    Returns:
        Dictionary mapping metric names to calculated values.
    """
    pass
```

### Commit Messages

Follow conventional commit format without decoration:

```text
❌ BAD:
🎉 feat(api): add new endpoint! Super excited about this! 🚀

✓ GOOD:
feat(api): add endpoint for user profile retrieval

Implements GET /api/v1/users/{id}/profile with caching support.
Includes validation for user ID format and authorization checks.
```

### Log Messages

Maintain professional, actionable log output:

```python
# ❌ BAD: Informal, decorated logs
logger.info("🎊 Woohoo! Successfully connected to database! 🎊")
logger.error("💥 Uh oh! Something went wrong! 😱")

# ✓ GOOD: Professional, informative logs
logger.info("Database connection established to postgresql://prod-db:5432")
logger.error("Database connection failed: timeout after 30s")
```

---

## Code Readability Guidelines

### Variable and Function Names

Use clear, descriptive names that reflect business or technical concepts:

```python
# ❌ BAD: Cutesy or informal naming
def fetchyFetch() -> None:
    magic_number = 42  # lol random
    super_duper_list = []

# ✓ GOOD: Professional, descriptive naming
def fetch_remote_data() -> None:
    default_timeout_seconds = 42
    validation_results = []
```

### Comments

Comments should explain **why**, not **what**:

```python
# ❌ BAD: Obvious comment with decoration
# ✨ Loop through all items ✨
for item in items:
    process(item)

# ✓ GOOD: Explains rationale when needed
# Process items sequentially to maintain transaction order
for item in items:
    process(item)
```

### Code Structure

Let the code speak for itself through clear structure:

```python
# ❌ BAD: Over-commented with decorative separators
# ==========================================
# 🌟 MAIN PROCESSING SECTION 🌟
# ==========================================
def main() -> None:
    # -------- Step 1: Initialize --------
    setup()
    # -------- Step 2: Process --------
    process()

# ✓ GOOD: Clear structure without decoration
def main() -> None:
    """Execute the main processing workflow."""
    setup()
    process()
```

---

## Exceptions

There are **no exceptions** to the no-emoji rule in production code or technical documentation. This applies regardless of:

- Target audience (internal, external, beginner, expert)
- Type of project (open source, proprietary, personal)
- Communication channel (code, docs, issues, PRs)
- Programming language or technology stack

---

## Rationale

### Why No Emojis?

1. **Accessibility**: Screen readers handle emojis inconsistently
2. **Searchability**: Emojis complicate text search and grep operations
3. **Professionalism**: Technical documentation demands clarity over decoration
4. **Compatibility**: Not all terminals, editors, or tools render emojis correctly
5. **Internationalization**: Emoji meanings vary across cultures
6. **Version control**: Emojis can cause encoding issues in diffs and logs
7. **Parsing**: Code analysis tools may choke on non-ASCII characters

### Human-Like Does Not Mean Casual

"Human-like" means writing as a **professional engineer** would:

- Clear and direct communication
- Technical precision without jargon abuse
- Appropriate formality for technical context
- Focus on information density and accuracy

It does **not** mean:

- Casual or conversational tone
- Emotional language or enthusiasm markers
- Decorative elements or visual flair
- Informal abbreviations or slang

---

## Checklist

Before committing code or documentation, verify:

- [ ] No emojis in source files
- [ ] No emojis in documentation
- [ ] No emojis in commit messages
- [ ] No decorative ASCII art or separators (unless required by format, e.g., structured comments)
- [ ] No informal language in technical documentation
- [ ] No unnecessary exclamation marks or emotional language
- [ ] Comments explain rationale, not obvious operations
- [ ] Log messages are actionable and informative
- [ ] Names are descriptive and professional

---

## Summary

Write code and documentation that you would be proud to present in a code review with senior engineers. Maintain technical precision, clarity, and professionalism throughout all artifacts. Let the quality of your logic and architecture speak louder than decorative elements ever could.
