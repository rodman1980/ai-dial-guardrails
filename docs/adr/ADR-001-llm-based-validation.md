# ADR-001: LLM-Based Validation Over Regex

## Status
Accepted (2025-12-31)

## Context

The project needs input validation to detect prompt injection attempts before sending queries to the main generative LLM. Two primary approaches were considered:

1. **Regex/Rule-Based**: Pattern matching for known attack signatures
2. **LLM-Based**: Using a dedicated validator LLM to analyze queries

### Problem Statement

Prompt injection attacks are sophisticated and constantly evolving. Attack patterns include:
- Instruction override ("ignore previous instructions")
- Jailbreak attempts (roleplay, pretend scenarios)
- Structured injection (JSON, XML, SQL templates)
- Many-shot attacks (pattern establishment)
- Context window saturation
- Social engineering

Static regex patterns struggle with:
- Natural language variation and paraphrasing
- New attack vectors not yet documented
- Context-dependent threats (legitimate vs. malicious intent)
- Adversarial evasion (obfuscation, encoding)

### Educational Context

This is an **educational project** demonstrating guardrail strategies. The goal is to show layered defense patterns rather than achieve production-level security.

## Decision

**We will use LLM-based input validation as the primary guardrail mechanism.**

Implementation:
- Dedicated validator LLM analyzes user queries using a security-focused prompt
- Validator returns structured output (Pydantic `ValidationResult`) with `is_valid` flag and `reason`
- Only queries passing validation proceed to the main generative LLM
- Failed validations are rejected with explanations for debugging

Pattern:
```python
prompt_template → validator_llm → PydanticOutputParser → ValidationResult
```

## Consequences

### Positive

1. **Adaptability**: LLM understands context and semantic intent, catching paraphrased attacks
2. **Explainability**: Validation provides human-readable `reason` field for debugging
3. **Extensibility**: Update detection logic by modifying the validation prompt (no code changes)
4. **Coverage**: Catches novel attack patterns not yet documented in regex rules
5. **Educational Value**: Demonstrates LLM-as-judge pattern for security applications

### Negative

1. **Latency**: Adds ~1-2 seconds per query (extra LLM call)
2. **Cost**: Doubles LLM API calls (validator + main generation)
3. **Reliability**: Validator itself can be tricked by adversarial inputs (second-order attacks)
4. **Complexity**: Requires prompt engineering for validator; sensitive to prompt quality
5. **False Positives**: Overly conservative validator may block legitimate queries

### Trade-offs Accepted

- **Latency vs. Security**: Accept extra latency for better attack detection
- **Cost vs. Coverage**: Accept higher API costs for broader protection
- **Simplicity vs. Robustness**: Accept complexity of LLM-based validation for better robustness

## Alternatives Considered

### Alternative 1: Pure Regex/Rule-Based Validation

**Approach**: Maintain a list of regex patterns for known attack signatures.

**Pros**:
- Fast (< 1ms)
- Deterministic and testable
- No API costs
- Fully offline

**Cons**:
- High maintenance (constant pattern updates)
- Brittle (easily bypassed via paraphrasing)
- Low coverage (misses novel attacks)
- High false negatives (attacks slip through)

**Reason for Rejection**: Insufficient for educational demonstration; shows limitations of static rules.

### Alternative 2: Hybrid (Regex Pre-filter + LLM Fallback)

**Approach**: Run fast regex checks first; only call LLM validator for ambiguous cases.

**Pros**:
- Optimized latency (most queries handled by regex)
- Lower API costs
- Combines determinism and adaptability

**Cons**:
- Increased complexity (two validation paths)
- Still vulnerable at regex layer
- Harder to test and maintain

**Reason for Rejection**: Over-engineered for educational scope; complicates learning objectives.

### Alternative 3: Embedding-Based Similarity Detection

**Approach**: Embed queries and compare to known attack vectors using cosine similarity.

**Pros**:
- Fast inference (after initial embedding)
- Handles paraphrasing well
- No LLM call needed (offline after model load)

**Cons**:
- Requires training/fine-tuning attack corpus
- Less explainable (similarity scores, no reasons)
- Harder to update (requires retraining)
- Complex setup for educational project

**Reason for Rejection**: Too advanced for educational scope; reduces clarity.

### Alternative 4: No Input Validation (System Prompt Only)

**Approach**: Rely solely on system prompt constraints and LLM fine-tuning.

**Pros**:
- Zero latency overhead
- Simplest implementation
- Demonstrates LLM instruction-following

**Cons**:
- Highly vulnerable to prompt injection
- No defense-in-depth
- Poor educational value (shows vulnerability, not mitigation)

**Reason for Rejection**: Defeats project purpose of demonstrating guardrails.

## Implementation Notes

### Validation Prompt Template

The validator uses a security-focused prompt (`VALIDATION_PROMPT_TEMPLATE`) that:
- Defines threat categories (instruction override, jailbreak, structured injection, etc.)
- Instructs LLM to analyze query for red flags
- Requests structured JSON output via `PydanticOutputParser`

### Fail-Safe Behavior

On `OutputParserException` (LLM response cannot be parsed):
- Default to **rejection** (fail-safe)
- Return `ValidationResult(is_valid=False, reason="Parse error")`

This conservative approach prioritizes security over availability.

### Testing Strategy

- Unit tests for known attack patterns from [PROMPT_INJECTIONS_TO_TEST.md](../../tasks/PROMPT_INJECTIONS_TO_TEST.md)
- Measure false positive rate on legitimate queries
- Measure false negative rate on documented attacks
- Adversarial testing with novel attack variations

## Future Considerations

### Potential Improvements

1. **Caching**: Cache validation results for identical/similar queries
2. **Rate Limiting**: Limit validator LLM calls to prevent DoS
3. **Ensemble**: Combine regex pre-filter with LLM for cost optimization
4. **Fine-Tuning**: Train custom validator model on attack corpus
5. **Monitoring**: Track validation metrics (rejection rate, latency, false positives)

### Supersession Triggers

This ADR may be superseded if:
- Regex patterns achieve > 95% detection rate in testing
- Production latency requirements become critical (< 500ms)
- Validator LLM shows unacceptable false positive rate (> 10%)
- Cost constraints prohibit dual LLM calls

## References

- [OWASP LLM Top 10: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Anthropic: Constitutional AI](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback)
- [LangChain: Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- Task Implementation: [tasks/t_2/input_llm_based_validation.py](../../tasks/t_2/input_llm_based_validation.py)

---

**Related ADRs**:
- [ADR-003: Layered Defense Strategy](./ADR-003-layered-defense.md) - Multi-layer security architecture
