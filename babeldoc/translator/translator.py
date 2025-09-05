import contextlib
import json
import logging
import threading
import time
import unicodedata
from abc import ABC
from abc import abstractmethod

import httpx
import openai
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from babeldoc.translator.cache import TranslationCache
from babeldoc.utils.atomic_integer import AtomicInteger

logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    """
    A rate limiter using the leaky bucket algorithm to ensure a smooth, constant rate of requests.
    This implementation is thread-safe and robust against system clock changes.
    """

    def __init__(self, max_qps: int):
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.lock = threading.Lock()
        # Use monotonic time to prevent issues with system time changes
        self.next_request_time = time.monotonic()

    def wait(self, _rate_limit_params: dict = None):
        """
        Blocks until the next request can be processed, ensuring the rate limit is not exceeded.
        """
        with self.lock:
            now = time.monotonic()

            wait_duration = self.next_request_time - now
            if wait_duration > 0:
                time.sleep(wait_duration)

            # Update the next allowed request time.
            # If the limiter has been idle, the next request should start from 'now'.
            now = time.monotonic()
            self.next_request_time = (
                max(self.next_request_time, now) + self.min_interval
            )

    def set_max_qps(self, max_qps: int):
        """
        Updates the maximum queries per second. This operation is thread-safe.
        """
        if max_qps <= 0:
            raise ValueError("max_qps must be a positive number")
        with self.lock:
            self.max_qps = max_qps
            self.min_interval = 1.0 / max_qps


_translate_rate_limiter = RateLimiter(5)


def set_translate_rate_limiter(max_qps):
    _translate_rate_limiter.set_max_qps(max_qps)


class BaseTranslator(ABC):
    # Due to cache limitations, name should be within 20 characters.
    # cache.py: translate_engine = CharField(max_length=20)
    name = "base"
    lang_map = {}

    def __init__(self, lang_in, lang_out, ignore_cache):
        self.ignore_cache = ignore_cache
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
            },
        )

        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )
            logger.info(
                f"{self.name} translate cache call count: {self.translate_cache_call_count}",
            )

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def get_cache_key(self, text):
        """
        Generate cache key for the given text. Override in subclasses to include additional parameters.
        :param text: text to translate
        :return: cache key string
        """
        return text

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        cache_key = self.get_cache_key(text)
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(cache_key)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        _translate_rate_limiter.wait()
        translation = self.do_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(cache_key, translation)
        return translation

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        cache_key = self.get_cache_key(text)
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(cache_key)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        _translate_rate_limiter.wait()
        translation = self.do_llm_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(cache_key, translation)
        return translation

    @abstractmethod
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int):
        return self.get_rich_text_left_placeholder(placeholder_id)


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        ignore_cache=False,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
                timeout=60,  # Set a reasonable timeout
            ),
        )
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def prompt(self, text):
        return [
            {
                "role": "system",
                "content": "You are a professional,authentic machine translation engine.",
            },
            {
                "role": "user",
                "content": f";; Treat next line as plain text input and translate it into {self.lang_out}, output translation ONLY. If translation is unnecessary (e.g. proper nouns, codes, {'{{1}}, etc. '}), return the original text. NO explanations. NO notes. Input:\n\n{text}",
            },
        ]

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )
        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def update_token_count(self, response):
        try:
            if response.usage and response.usage.total_tokens:
                self.token_count.inc(response.usage.total_tokens)
            if response.usage and response.usage.prompt_tokens:
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if response.usage and response.usage.completion_tokens:
                self.completion_token_count.inc(response.usage.completion_tokens)
        except Exception as e:
            logger.exception("Error updating token count")

    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"
        return "{{" + str(placeholder_id) + "}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"


class QwenMTTranslator(BaseTranslator):
    name = "qwenmt"

    # Language code mapping for QwenMT API
    lang_map = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "zh-hans": "Chinese",
        "zh-tw": "Traditional Chinese",
        "zh-hant": "Traditional Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "th": "Thai",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        "hi": "Hindi",
        "tr": "Turkish",
        "pl": "Polish",
        "nl": "Dutch",
    }

    def __init__(
        self,
        lang_in,
        lang_out,
        qwenmt_model="qwen-mt-turbo",
        qwenmt_plus_model="qwen-plus-latest",
        base_url=None,
        api_key=None,
        ignore_cache=False,
    ):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.qwenmt_model = qwenmt_model
        self.qwenmt_plus_model = qwenmt_plus_model

        self.client = openai.OpenAI(
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=None
                ),
                timeout=60,
            ),
        )

        self.add_cache_impact_parameters("qwenmt_model", self.qwenmt_model)
        self.add_cache_impact_parameters("qwenmt_plus_model", self.qwenmt_plus_model)
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

        # Convert language codes to QwenMT format
        self.source_lang = self.lang_map.get(lang_in.lower(), lang_in)
        self.target_lang = self.lang_map.get(lang_out.lower(), lang_out)

        # Cache for single glossary
        self._current_glossary = None

    def set_glossaries(self, glossaries):
        """Set glossary for translation. Only the first glossary will be used."""
        if glossaries and len(glossaries) > 0:
            self._current_glossary = glossaries[0]
        else:
            self._current_glossary = None

    def get_cache_key(self, text):
        """
        Generate cache key including glossary information for QwenMT.
        :param text: text to translate
        :return: cache key string including glossary state
        """
        if not self._current_glossary:
            return text

        # Get active terms for the current text
        terms = self._get_active_terms_for_text(text)
        if not terms:
            return text

        # Serialize terms to create a consistent cache key
        terms_json = json.dumps(terms, sort_keys=True, ensure_ascii=False)
        return f"{terms_json}|{text}"

    def _get_active_terms_for_text(self, text):
        """Get only the glossary terms that appear in the given text"""
        terms = []
        if self._current_glossary:
            # Get active entries for the current text from single glossary
            active_entries = self._current_glossary.get_active_entries_for_text(text)
            for source_term, target_term in active_entries:
                terms.append({"source": source_term, "target": target_term})
        return terms

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        """Use QwenMT for regular translation"""
        translation_options = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }

        # Add terms if glossary provided - only include terms found in the text
        if self._current_glossary:
            terms = self._get_active_terms_for_text(text)
            if terms:
                translation_options["terms"] = terms

        messages = [{"role": "user", "content": text}]

        response = self.client.chat.completions.create(
            model=self.qwenmt_model,
            messages=messages,
            extra_body={"translation_options": translation_options},
        )

        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """Use Qwen-Plus for complex LLM translation"""
        if text is None:
            return None

        messages = [{"role": "user", "content": text}]

        response = self.client.chat.completions.create(
            model=self.qwenmt_plus_model,
            messages=messages,
            max_tokens=2048,
            temperature=0,
        )

        self.update_token_count(response)
        return response.choices[0].message.content.strip()

    def update_token_count(self, response):
        """Update token count statistics"""
        try:
            if response.usage and response.usage.total_tokens:
                self.token_count.inc(response.usage.total_tokens)
            if response.usage and response.usage.prompt_tokens:
                self.prompt_token_count.inc(response.usage.prompt_tokens)
            if response.usage and response.usage.completion_tokens:
                self.completion_token_count.inc(response.usage.completion_tokens)
        except Exception as e:
            logger.exception("Error updating token count")

    def get_formular_placeholder(self, placeholder_id: int):
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return "</style>", r"<\s*\/\s*style\s*>"
