import html
import json
import logging
import re
import threading
import time
from typing import Dict
import unicodedata
from abc import ABC
from pydantic import BaseModel
import openai
import requests

from babeldoc.document_il.translator.cache import TranslationCache

logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class RateLimiter:
    def __init__(self, max_qps: int):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.last_requests = []  # Track last N requests
        self.window_size = max_qps  # Track requests in a sliding window
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()

            # Clean up old requests outside the 1-second window
            while self.last_requests and now - self.last_requests[0] > 1.0:
                self.last_requests.pop(0)

            # If we have less than max_qps requests in the last second, allow immediately
            if len(self.last_requests) < self.max_qps:
                self.last_requests.append(now)
                return

            # Otherwise, wait until we can make the next request
            next_time = self.last_requests[0] + 1.0
            if next_time > now:
                time.sleep(next_time - now)
            self.last_requests.pop(0)
            self.last_requests.append(next_time)

    def set_max_qps(self, max_qps):
        self.max_qps = max_qps
        self.min_interval = 1.0 / max_qps
        self.window_size = max_qps


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
        logger.info(f"{self.name} translate call count: {self.translate_call_count}")
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

    def translate(self, text, ignore_cache=False):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_translate(text)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def do_translate(self, text):
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


class GoogleTranslator(BaseTranslator):
    name = "google"
    lang_map = {"zh": "zh-CN"}

    def __init__(self, lang_in, lang_out, ignore_cache=False):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "http://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)",
        }

    def do_translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.endpoint,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<',
            response.text,
        )
        if response.status_code == 400:
            result = "IRREPARABLE TRANSLATION ERROR"
        else:
            response.raise_for_status()
            result = html.unescape(re_result[0])
        return remove_control_characters(result)


class BingTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "bing"
    lang_map = {"zh": "zh-Hans"}

    def __init__(self, lang_in, lang_out, ignore_cache=False):
        super().__init__(lang_in, lang_out, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://www.bing.com/translator"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        }

    def find_sid(self):
        response = self.session.get(self.endpoint)
        response.raise_for_status()
        url = response.url[:-10]
        ig = re.findall(r"\"ig\":\"(.*?)\"", response.text)[0]
        iid = re.findall(r"data-iid=\"(.*?)\"", response.text)[-1]
        key, token = re.findall(
            r"params_AbusePreventionHelper\s=\s\[(.*?),\"(.*?)\",",
            response.text,
        )[0]
        return url, ig, iid, key, token

    def do_translate(self, text):
        text = text[:1000]  # bing translate max length
        url, ig, iid, key, token = self.find_sid()
        response = self.session.post(
            f"{url}ttranslatev3?IG={ig}&IID={iid}",
            data={
                "fromLang": self.lang_in,
                "to": self.lang_out,
                "text": text,
                "token": token,
                "key": key,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]


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
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.model = model
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))

    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
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

    def get_formular_placeholder(self, placeholder_id: int):
        return "{{v" + str(placeholder_id) + "}}"
        return "{{" + str(placeholder_id) + "}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        return self.get_formular_placeholder(placeholder_id)

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        return self.get_formular_placeholder(placeholder_id + 1)


class OpenAIAdvancedTranslator(OpenAITranslator):
    name = "openai_advanced"

    lang_map = {
        "en": "英语",
        "en-US": "英语",
        "zh-CN": "简体中文",
        "zh": "简体中文",
        "zh-TW": "繁体中文",
        "ja": "日语",
        "ko": "韩语",
        "ru": "俄语",
        "fr": "法语",
        "de": "德语",
        "it": "意大利语",
        "es": "西班牙语",
    }

    class TranslationOutput(BaseModel):
        translated: str  # 译文

    # 获取模型的 JSON Schema
    json_schema = TranslationOutput.model_json_schema()

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        ignore_cache=False,
        qps: int = 200,
    ):
        # 确保语言代码在映射表中存在
        if lang_in != "auto" and lang_in not in self.lang_map:
            raise ValueError(f"不支持的源语言: {lang_in}")
        if lang_out not in self.lang_map:
            raise ValueError(f"不支持的目标语言: {lang_out}")

        super().__init__(
            lang_in=lang_in,
            lang_out=lang_out,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        set_translate_rate_limiter(qps)

    def translate(self, text, ignore_cache=False, dictionary: Dict[str, str] = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                self.translate_cache_call_count += 1
                return cache
        _translate_rate_limiter.wait()
        translation = self.do_translate(text, dictionary)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def do_translate(self, text, dictionary: Dict[str, str] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, dictionary),
            # extra_body={"guided_json": self.json_schema},  # add
        )
        # result = self.TranslationOutput.model_validate_json(  # add
        #     response.choices[0].message.content  # add
        # )
        # return result.translated.strip()  # modify
        return response.choices[0].message.content

    def prompt(self, text, dictionary: Dict[str, str] = None):
        is_auto_lang = self.lang_in == "auto"
        in_lang_part = "任何" if is_auto_lang else f"{self.lang_map[self.lang_in]}"
        # 生成非目标语言处理说明
        out_lang_part = (
            f"{self.lang_map[self.lang_out]}"
            if is_auto_lang
            else f"{self.lang_map[self.lang_out]}, 源文本中非{self.lang_map[self.lang_in]}的部分内容直接使用原文作为译文"
        )
        if dictionary:
            dictionary_part = "\n\n参考术语:\n" + "\n".join(
                f"{k}: {v}" for k, v in dictionary.items()
            )
        else:
            dictionary_part = ""
        result = [
            {
                "role": "system",
                "content": rf"""你是一位专业的多语言法律领域翻译专家. 请遵循以下指南:
1. 翻译原则
- 严格遵循法律用语的专业性和严谨性
- 准确传达法律条款的权利义务关系
- 保持法律术语的规范性和一致性
- 确保译文符合目标语言的法律表述习惯

2. 基本要求
- 严格保持原文的格式,标点和段落结构
- 保留所有数学公式,代码等特殊标记
- 使用权威法律词典和判例中的标准译法
- 在保证法律含义准确的前提下使译文通顺
- 对合同主体,权利义务,期限等关键内容的翻译尤其谨慎

3. 特殊情况处理
- 遇到不确定或多种译法的术语,选择最合适的译法
- 遇到文化差异内容和语气词,使用目标语言的习惯表达
- 遇到短词组或单个词语,如无上下文,选择最常用的译法
- 遇到短文本,如无上下文,选择最常用的译法{dictionary_part}
""",
            },
            {
                "role": "user",
                "content": f";; 将用户在下一行输入的{in_lang_part}内容翻译成{out_lang_part}.仅输出译文,如果是不必要的翻译(例如专有名词、代码、{'{{1}}, 等'}),返回原文.不要解释,不要备注.输入内容:",
            },
            {
                "role": "user",
                "content": text,
            },
        ]
        # print(result)
        return result


if __name__ == "__main__":
    translator = OpenAIAdvancedTranslator(
        lang_in="zh-CN",
        lang_out="en-US",
        # model="qwen2-instruct",
        model="Qwen/Qwen2.5-72B-Instruct",
        # base_url="http://127.0.0.1:9997/v1",
        base_url="https://api.siliconflow.cn/v1",
        # api_key="EMPTY",
        api_key="sk-kxyyqvlclkbswvrnzyzjdzxoarunqjunjylvdeutleaxwhoi",
    )
    texts = [
        "与",
        "关于",
        "之",
        "定义",
        "终止",
        "兹此",
        "用途",
        "B",
        "C",
        "G",
        "J",
        "元",
        "啊",
        "软件许可所有权",
        "甲方",
        "乙方",
        "丙方",
        "丁方",
        "我是经过 UFCC 许可认证的。",
        ", You have no permission",
        ", You have no permission {1}",
        "你没有许可 {1}",
    ]
    for text in texts:
        print(
            translator.translate(
                text,
                dictionary={
                    "与": "and",
                    "关于": "about",
                    "之": "of",
                },
            )
        )

    # 测试临时词典
