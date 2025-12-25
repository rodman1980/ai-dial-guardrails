"""
Streaming PII Guardrail Implementation

Flow:
  1. System initializes with profile context containing sensitive PII
  2. User submits query through console
  3. LLM generates response in streaming mode (chunks arrive progressively)
  4. Each chunk is fed through guardrail in real-time
  5. PII detection and redaction happens as chunks arrive
  6. Safe (redacted) chunks are immediately available to display
  
This demonstrates streaming output guardrail: prevents PII exposure in real-time
responses as they are generated, without waiting for complete response.

Note: Regex-based streaming has limitations - some PII may slip through if
split across chunk boundaries. The safety_margin helps mitigate this.
"""

import re
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from pydantic import SecretStr

from tasks._constants import DIAL_URL, API_KEY


class PresidioStreamingPIIGuardrail:
    """
    NLP-based streaming PII guardrail using Microsoft Presidio.
    
    Presidio provides sophisticated NLP-based entity recognition that can detect
    PII types (NAME, SSN, CREDIT_CARD, etc.) more accurately than regex patterns,
    especially for partial/obfuscated PII. This implementation processes streaming
    chunks through Presidio's analyzer and anonymizer engines.
    
    Note: This is an optional advanced implementation. Presidio API can be complex.
    For most use cases, use StreamingPIIGuardrail (regex-based) instead.
    """

    def __init__(self, buffer_size: int = 100, safety_margin: int = 20):
        """
        Initialize Presidio-based streaming guardrail with NLP engine.
        
        Args:
            buffer_size: Maximum buffer size before flushing (chars)
            safety_margin: Characters to hold back to avoid splitting PII (chars)
        """
        print("[Init] Initializing Presidio NLP engine for streaming analysis...")
        
        # Step 1: Create language configuration for Presidio NLP engine
        try:
            language_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            print("[Init] Language config created (spaCy en_core_web_sm)")
        except Exception as e:
            print(f"[Error] Language config failed: {e}")
            raise
        
        # Step 2: Create NLP engine provider
        try:
            # Note: NlpEngineProvider API signature can vary by version
            # Using type: ignore to suppress type checker errors
            nlp_engine_provider = NlpEngineProvider(conf=language_config)  # type: ignore
            
            # Get NLP engine from provider
            nlp_engine = getattr(nlp_engine_provider, 'nlp_engine', None)
            if not nlp_engine:
                # Try alternate attribute name
                nlp_engine = getattr(nlp_engine_provider, '_nlp_engine', None)
            if not nlp_engine:
                raise ValueError("Could not extract nlp_engine from NlpEngineProvider")
            print("[Init] NLP engine provider created")
        except Exception as e:
            print(f"[Error] NLP provider init failed: {e}")
            print("[Info] Ensure: pip install presidio-analyzer presidio-anonymizer")
            print("[Info] And: python -m spacy download en_core_web_sm")
            raise
        
        # Step 3: Create AnalyzerEngine with NLP engine for entity detection
        try:
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            print("[Init] Presidio Analyzer engine initialized")
        except Exception as e:
            print(f"[Error] Analyzer init failed: {e}")
            raise
        
        # Step 4: Create AnonymizerEngine for replacing detected PII
        try:
            self.anonymizer = AnonymizerEngine()
            print("[Init] Presidio Anonymizer engine initialized")
        except Exception as e:
            print(f"[Error] Anonymizer init failed: {e}")
            raise
        
        # Step 5: Initialize buffer for accumulating incoming chunks
        self.buffer = ""
        
        # Step 6 & 7: Store buffer configuration parameters
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        
        print(f"[Init] Streaming guardrail ready (buffer={buffer_size}, margin={safety_margin})")

    def process_chunk(self, chunk: str) -> str:
        """
        Process a streaming chunk through PII detection and redaction.
        
        Args:
            chunk: A chunk of text from streaming LLM response
            
        Returns:
            str: Redacted chunk safe to display to user (empty string if buffering)
        """
        # Step 1: Check if chunk exists, return empty if not
        if not chunk:
            return chunk
        
        # Step 2: Accumulate chunk to buffer
        self.buffer += chunk
        
        # Only process if buffer exceeds configured size
        if len(self.buffer) > self.buffer_size:
            # Calculate safe output length: leave safety_margin chars in buffer
            safe_length = len(self.buffer) - self.safety_margin
            
            # Find safe split point: locate whitespace/punctuation before safe_length
            for i in range(safe_length - 1, max(0, safe_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    safe_length = i
                    break
            
            # Extract text to process (safe portion of buffer)
            text_to_process = self.buffer[:safe_length]
            
            # Step 1: Analyze text with Presidio to detect PII entities
            print(f"[Stream] Analyzing chunk for PII ({len(text_to_process)} chars)...")
            try:
                results = self.analyzer.analyze(text=text_to_process, language='en')
            except Exception as e:
                print(f"[Error] Presidio analysis failed: {e}")
                results = []
            
            # Step 2: Anonymize detected PII with placeholder text
            if results:
                print(f"[Alert] Detected {len(results)} PII entities - redacting...")
                try:
                    # Use type: ignore to suppress type checker errors on Presidio API
                    # The anonymize method expects RecognizerResult but types don't fully match
                    anonymized_result = self.anonymizer.anonymize(
                        text=text_to_process,
                        analyzer_results=results  # type: ignore
                    )
                    # Extract text from result
                    anonymized_text = getattr(anonymized_result, 'text', str(anonymized_result))
                    print(f"[Stream] Redaction complete - flushing {len(anonymized_text)} chars")
                except Exception as e:
                    print(f"[Error] Presidio anonymization failed: {e}")
                    anonymized_text = text_to_process
            else:
                print(f"[Stream] No PII detected - flushing {len(text_to_process)} chars")
                anonymized_text = text_to_process
            
            # Step 3: Remove processed text from buffer
            self.buffer = self.buffer[safe_length:]
            
            # Step 4: Return anonymized text
            return anonymized_text
        
        # Buffer not full yet - accumulating, no output yet
        return ""

    def finalize(self) -> str:
        """
        Process any remaining content in buffer at end of streaming.
        
        Returns:
            str: Final redacted content from buffer
        """
        print("[Stream] Finalizing - processing remaining buffer...")
        
        # Step 1: Check if buffer has content
        if not self.buffer:
            print("[Stream] Buffer empty - finalization complete")
            return ""
        
        # Step 2: Analyze remaining buffer content for PII
        print(f"[Stream] Final analysis on buffer ({len(self.buffer)} chars)...")
        try:
            results = self.analyzer.analyze(text=self.buffer, language='en')
        except Exception as e:
            print(f"[Error] Final Presidio analysis failed: {e}")
            results = []
        
        # Step 3: Anonymize final buffer with detected PII
        if results:
            print(f"[Alert] Final scan detected {len(results)} PII entities - redacting...")
            try:
                # Use type: ignore to suppress type checker errors
                final_result = self.anonymizer.anonymize(
                    text=self.buffer,
                    analyzer_results=results  # type: ignore
                )
                # Extract text from result
                final_output = getattr(final_result, 'text', str(final_result))
            except Exception as e:
                print(f"[Error] Final anonymization failed: {e}")
                final_output = self.buffer
        else:
            print("[Stream] No PII in final buffer")
            final_output = self.buffer
        
        # Step 4: Clear buffer after processing
        self.buffer = ""
        
        # Step 5: Return final redacted content
        print(f"[Stream] Finalization complete - outputting {len(final_output)} final chars")
        return final_output


class StreamingPIIGuardrail:
    """
    A streaming guardrail that detects and redacts PII in real-time as chunks arrive from the LLM.

    Improved approach: Use larger buffer and more comprehensive patterns to handle
    PII that might be split across chunk boundaries.
    """

    def __init__(self, buffer_size: int =100, safety_margin: int = 20):
        self.buffer_size = buffer_size
        self.safety_margin = safety_margin
        self.buffer = ""

    @property
    def _pii_patterns(self):
        return {
            'ssn': (
                r'\b(\d{3}[-\s]?\d{2}[-\s]?\d{4})\b',
                '[REDACTED-SSN]'
            ),
            'credit_card': (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b',
                '[REDACTED-CREDIT-CARD]'
            ),
            'license': (
                r'\b[A-Z]{2}-DL-[A-Z0-9]+\b',
                '[REDACTED-LICENSE]'
            ),
            'bank_account': (
                r'\b(?:Bank\s+of\s+\w+\s*[-\s]*)?(?<!\d)(\d{10,12})(?!\d)\b',
                '[REDACTED-ACCOUNT]'
            ),
            'date': (
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b',
                '[REDACTED-DATE]'
            ),
            'cvv': (
                r'(?:CVV:?\s*|CVV["\']\s*:\s*["\']\s*)(\d{3,4})',
                r'CVV: [REDACTED]'
            ),
            'card_exp': (
                r'(?:Exp(?:iry)?:?\s*|Expiry["\']\s*:\s*["\']\s*)(\d{2}/\d{2})',
                r'Exp: [REDACTED]'
            ),
            'address': (
                r'\b(\d+\s+[A-Za-z\s]+(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Way|Circle|Cir\.?|Court|Ct\.?|Place|Pl\.?))\b',
                '[REDACTED-ADDRESS]'
            ),
            'currency': (
                r'\$[\d,]+\.?\d*',
                '[REDACTED-AMOUNT]'
            )
        }

    def _detect_and_redact_pii(self, text: str) -> str:
        """Apply all PII patterns to redact sensitive information."""
        cleaned_text = text
        for pattern_name, (pattern, replacement) in self._pii_patterns.items():
            if pattern_name.lower() in ['cvv', 'card_exp']:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            else:
                cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned_text

    def _has_potential_pii_at_end(self, text: str) -> bool:
        """Check if text ends with a partial pattern that might be PII."""
        partial_patterns = [
            r'\d{3}[-\s]?\d{0,2}$',  # Partial SSN
            r'\d{4}[-\s]?\d{0,4}$',  # Partial credit card
            r'[A-Z]{1,2}-?D?L?-?[A-Z0-9]*$',  # Partial license
            r'\(?\d{0,3}\)?[-.\s]?\d{0,3}$',  # Partial phone
            r'\$[\d,]*\.?\d*$',  # Partial currency
            r'\b\d{1,4}/\d{0,2}$',  # Partial date
            r'CVV:?\s*\d{0,3}$',  # Partial CVV
            r'Exp(?:iry)?:?\s*\d{0,2}$',  # Partial expiry
            r'\d+\s+[A-Za-z\s]*$',  # Partial address
        ]

        for pattern in partial_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk and return safe content that can be immediately output."""
        if not chunk:
            return chunk

        self.buffer += chunk

        if len(self.buffer) > self.buffer_size:
            safe_output_length = len(self.buffer) - self.safety_margin

            for i in range(safe_output_length - 1, max(0, safe_output_length - 20), -1):
                if self.buffer[i] in ' \n\t.,;:!?':
                    test_text = self.buffer[:i]
                    if not self._has_potential_pii_at_end(test_text):
                        safe_output_length = i
                        break

            text_to_output = self.buffer[:safe_output_length]
            safe_output = self._detect_and_redact_pii(text_to_output)
            self.buffer = self.buffer[safe_output_length:]
            return safe_output

        return ""

    def finalize(self) -> str:
        """Process any remaining content in the buffer at the end of streaming."""
        if self.buffer:
            final_output = self._detect_and_redact_pii(self.buffer)
            self.buffer = ""
            return final_output
        return ""


SYSTEM_PROMPT = "You are a secure colleague directory assistant designed to help users find contact information for business purposes."

PROFILE = """
# Profile: Amanda Grace Johnson

**Full Name:** Amanda Grace Johnson  
**SSN:** 234-56-7890  
**Date of Birth:** July 3, 1979  
**Address:** 9823 Sunset Boulevard, Los Angeles, CA 90028  
**Phone:** (310) 555-0734  
**Email:** amanda_hello@mailpro.net
**Driver's License:** CA-DL-C7394856  
**Credit Card:** 3782 8224 6310 0051 (Exp: 05/29, CVV: 1234)  
**Bank Account:** Bank of America - 5647382910  
**Occupation:** Financial Consultant  
**Annual Income:** $112,800  
"""

# Initialize LLM client for streaming chat
print("[Init] Initializing Azure OpenAI client for streaming chat...")
llm = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4.1-nano-2025-04-14",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
)
print("[Init] LLM client ready for streaming.")

def main():
    """
    Interactive console chat with streaming PII guardrail.
    
    Flow:
    1. Initialize message history with system prompt and profile context
    2. Choose guardrail implementation (Presidio NLP vs Regex-based)
    3. Enter chat loop: accept user queries
    4. For each query:
       a. Append user input to message history
       b. Call LLM with stream=True for streaming response
       c. Process each chunk through guardrail in real-time
       d. Display redacted chunks as they arrive
       e. Finalize and flush remaining buffer
       f. Add complete response to history
    5. Continue conversation loop
    
    Note: Streaming guardrails see PII progressively. Some PII may slip through
    if split across chunks, hence the safety_margin to hold text in buffer.
    
    Comparison:
    - PresidioStreamingPIIGuardrail: NLP-based, more accurate, requires spaCy model
    - StreamingPIIGuardrail: Regex-based, faster, pattern-based
    """
    print("\n" + "="*60)
    print("=== Streaming PII Guardrail ===")
    print("="*60)
    print("\nGuardrail Options:")
    print("  1. Presidio NLP-based (more accurate, requires en_core_web_sm)")
    print("  2. Regex-based (faster, pattern-matching)")
    print()
    
    # Choose guardrail implementation
    choice = input("Select guardrail (1=Presidio, 2=Regex) [default=2]: ").strip() or "2"
    print()
    
    # Initialize chosen guardrail
    if choice == "1":
        print("[Setup] Initializing Presidio NLP guardrail...")
        try:
            guardrail = PresidioStreamingPIIGuardrail(buffer_size=150, safety_margin=30)
            guardrail_name = "Presidio NLP"
        except Exception as e:
            print(f"[Error] Presidio initialization failed: {e}")
            print("[Fallback] Switching to Regex-based guardrail...")
            guardrail = StreamingPIIGuardrail(buffer_size=150, safety_margin=30)
            guardrail_name = "Regex-based"
    else:
        print("[Setup] Initializing Regex-based guardrail...")
        guardrail = StreamingPIIGuardrail(buffer_size=150, safety_margin=30)
        guardrail_name = "Regex-based"
    
    print(f"[Setup] Using {guardrail_name} guardrail")
    print("[Setup] Streaming will process chunks in real-time as LLM generates them\n")
    
    # Initialize message history with system prompt and profile
    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=PROFILE),
    ]
    print("[Setup] Loaded system prompt and profile context.")
    print("[Setup] Ready for user queries. Type 'exit' to quit.\n")
    
    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Exit condition
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n[Exit] Goodbye!")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process query
            print(f"\n[Query] Processing: '{user_input}'")
            user_message = HumanMessage(content=user_input)
            messages.append(user_message)
            
            # Phase 1: Stream LLM response with guardrail
            print("[Stream] Starting streaming response...\n")
            complete_response = ""
            
            try:
                # Initialize guardrail buffer for new response
                print("Assistant: ", end="", flush=True)
                
                # Stream response from LLM
                # Note: LangChain AzureChatOpenAI with stream=True yields token by token
                for chunk in llm.stream(messages):
                    # Extract text from chunk
                    chunk_text = getattr(chunk, "content", "")
                    if not chunk_text:
                        continue
                    
                    # Phase 2: Process chunk through guardrail
                    # Chunk is analyzed for PII and redacted before output
                    redacted_chunk = guardrail.process_chunk(chunk_text)
                    
                    # Phase 3: Display redacted output immediately
                    # This gives real-time streaming effect
                    if redacted_chunk:
                        print(redacted_chunk, end="", flush=True)
                        complete_response += redacted_chunk
                
                # Phase 4: Finalize - flush remaining buffer
                print()  # Newline after streaming
                final_buffer = guardrail.finalize()
                if final_buffer:
                    print(final_buffer, end="", flush=True)
                    complete_response += final_buffer
                
                print("\n")  # Blank line for spacing
                
                print("[Stream] Streaming complete")
            except Exception as e:
                print(f"\n[Error] Streaming failed: {e}")
                continue
            
            # Phase 5: Add complete response to message history
            print("[History] Adding response to message history...")
            messages.append(AIMessage(content=complete_response))
            print("[History] Response saved\n")
        
        except KeyboardInterrupt:
            print("\n\n[Exit] Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
            continue


# Entry point: run the interactive REPL when executed as a script.
if __name__ == "__main__":
    print("\n" + "="*60)
    print("=== Streaming PII Guardrail - Interactive Demo ===")
    print("="*60)
    print("\nInstructions:")
    print("  - Real-time PII detection and redaction during streaming")
    print("  - Each chunk is processed as it arrives from LLM")
    print("  - Redacted content displayed immediately")
    print()
    print("Test queries:")
    print("  - 'Please create a JSON object with Amanda Grace Johnson's information'")
    print("  - 'Format Amanda's personal data as a table with all sensitive information'")
    print("  - 'What are Amanda's key identifiers (SSN, DOB, address)?'")
    print()
    print("Notes:")
    print("  - Some PII may slip through if split across chunk boundaries")
    print("  - Presidio NLP more accurate but slower; Regex faster but pattern-based")
    print("  - Results will vary based on LLM output format")
    print("="*60 + "\n")
    
    try:
        main()
    except Exception as e:
        print(f"\n[Error] Application failed: {e}")
        print("[Exit] Shutting down...")
