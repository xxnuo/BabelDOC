import copy
import logging
import re
from pathlib import Path

import tiktoken
from tqdm import tqdm

from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfFont
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il.midend.il_translator import (
    DocumentTranslateTracker,
)
from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
from babeldoc.format.pdf.document_il.midend.il_translator import PageTranslateTracker
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor

logger = logging.getLogger(__name__)


class BatchParagraph:
    def __init__(
        self,
        paragraphs: list[PdfParagraph],
        page_tracker: PageTranslateTracker,
    ):
        self.paragraphs = paragraphs
        self.trackers = [page_tracker.new_paragraph() for _ in paragraphs]


class ILTranslatorQwenMT:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
        tokenizer=None,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.shared_context_cross_split_part = (
            translation_config.shared_context_cross_split_part
        )

        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.tokenizer = tokenizer

        # Cache glossaries at initialization
        self._cached_glossaries = (
            self.shared_context_cross_split_part.get_glossaries_for_translation(
                translation_config.auto_extract_glossary
            )
        )

        # Create fallback ILTranslator for complex paragraphs
        self.il_translator = ILTranslator(
            translate_engine=translate_engine,
            translation_config=translation_config,
            tokenizer=self.tokenizer,
        )
        self.il_translator.use_as_fallback = True

        # Validate QwenMT support
        try:
            self.translate_engine.do_llm_translate(None)
        except NotImplementedError as e:
            raise ValueError("QwenMT translator requires LLM support") from e

        # Set glossaries in QwenMT translator
        if hasattr(self.translate_engine, "set_glossaries"):
            self.translate_engine.set_glossaries(self._cached_glossaries)

        self.ok_count = 0
        self.fallback_count = 0
        self.total_count = 0

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def translate(self, docs: Document) -> None:
        tracker = DocumentTranslateTracker()
        self.mid = 0

        if not self.translation_config.shared_context_cross_split_part.first_paragraph:
            # Try to find the first title paragraph
            title_paragraph = self.find_title_paragraph(docs)
            self.translation_config.shared_context_cross_split_part.first_paragraph = (
                copy.deepcopy(title_paragraph)
            )
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph = copy.deepcopy(
                title_paragraph
            )
            if title_paragraph:
                logger.info(f"Found first title paragraph: {title_paragraph.unicode}")

        # count total paragraph
        total = sum(
            [
                len(
                    [
                        p
                        for p in page.pdf_paragraph
                        if p.debug_id is not None and p.unicode is not None
                    ]
                )
                for page in docs.page
            ]
        )
        translated_ids = set()
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with PriorityThreadPoolExecutor(
                max_workers=self.translation_config.pool_max_workers,
            ) as executor2:
                with PriorityThreadPoolExecutor(
                    max_workers=self.translation_config.pool_max_workers,
                ) as executor:
                    # Cross-page and cross-column processing uses LLM (Qwen-Plus)
                    self.process_cross_page_paragraph(
                        docs,
                        executor,
                        pbar,
                        tracker,
                        executor2,
                        translated_ids,
                    )
                    # Cross-column detection per page (after cross-page processing)
                    for page in docs.page:
                        self.process_cross_column_paragraph(
                            page,
                            executor,
                            pbar,
                            tracker,
                            executor2,
                            translated_ids,
                        )
                    # Regular paragraphs use QwenMT
                    for page in docs.page:
                        self.process_page(
                            page,
                            executor,
                            pbar,
                            tracker.new_page(),
                            executor2,
                            translated_ids,
                        )

        path = self.translation_config.get_working_file_path("translate_tracking.json")

        if self.translation_config.debug:
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())
        logger.info(
            f"QwenMT Translation completed. Total: {self.total_count}, Successful: {self.ok_count}, Fallback: {self.fallback_count}"
        )

    def _is_body_text_paragraph(self, paragraph: PdfParagraph) -> bool:
        """判断正文段落（当前仅 layout_label == 'text'）。

        Args:
            paragraph: PDF paragraph to check

        Returns:
            True if this is a body text paragraph, False otherwise
        """
        return paragraph.layout_label in (
            "text",
            "plain text",
            "paragraph_hybrid",
        )

    def _should_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        translated_ids: set[int] | None = None,
        require_body_text: bool = False,
    ) -> bool:
        """Check if a paragraph should be translated based on common filtering criteria.

        Args:
            paragraph: PDF paragraph to check
            translated_ids: Set of already translated paragraph IDs
            require_body_text: Whether to additionally check if paragraph is body text

        Returns:
            True if paragraph should be translated, False otherwise
        """
        # Basic validation checks
        if paragraph.debug_id is None or paragraph.unicode is None:
            return False

        # Check if already translated
        if translated_ids is not None and id(paragraph) in translated_ids:
            return False

        # CID paragraph check
        if is_cid_paragraph(paragraph):
            return False

        # Minimum length check
        if len(paragraph.unicode) < self.translation_config.min_text_length:
            return False

        # Body text check if requested
        if require_body_text and not self._is_body_text_paragraph(paragraph):
            return False

        return True

    def _filter_paragraphs(
        self,
        page: Page,
        translated_ids: set[int] | None = None,
        require_body_text: bool = False,
    ) -> list[PdfParagraph]:
        """Get list of paragraphs that should be translated from a page.

        Args:
            page: Page to get paragraphs from
            translated_ids: Set of already translated paragraph IDs
            require_body_text: Whether to filter for body text paragraphs only

        Returns:
            List of paragraphs that should be translated
        """
        return [
            paragraph
            for paragraph in page.pdf_paragraph
            if self._should_translate_paragraph(
                paragraph, translated_ids, require_body_text
            )
        ]

    def _build_font_maps(
        self, page: Page
    ) -> tuple[dict[str, PdfFont], dict[int, dict[str, PdfFont]]]:
        """Build font maps for a page.

        Args:
            page: The page to build font maps for

        Returns:
            Tuple of (page_font_map, page_xobj_font_map)
        """
        page_font_map = {}
        for font in page.pdf_font:
            page_font_map[font.font_id] = font

        page_xobj_font_map = {}
        for xobj in page.pdf_xobject:
            page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            for font in xobj.pdf_font:
                page_xobj_font_map[xobj.xobj_id][font.font_id] = font

        return page_font_map, page_xobj_font_map

    def process_cross_page_paragraph(
        self,
        docs: Document,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: DocumentTranslateTracker | None = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set[int] | None = None,
    ):
        """Process cross-page paragraphs using LLM (Qwen-Plus)"""
        self.translation_config.raise_if_cancelled()

        if tracker is None:
            tracker = DocumentTranslateTracker()

        if translated_ids is None:
            translated_ids = set()

        # Process adjacent page pairs
        for i in range(len(docs.page) - 1):
            page_curr = docs.page[i]
            page_next = docs.page[i + 1]

            # Find body text paragraphs in current page
            curr_body_paragraphs = self._filter_paragraphs(
                page_curr, translated_ids, require_body_text=True
            )

            # Find body text paragraphs in next page
            next_body_paragraphs = self._filter_paragraphs(
                page_next, translated_ids, require_body_text=True
            )

            # Get last paragraph from current page and first paragraph from next page
            if not curr_body_paragraphs or not next_body_paragraphs:
                continue

            last_curr_paragraph = curr_body_paragraphs[-1]
            first_next_paragraph = next_body_paragraphs[0]

            # Skip if either paragraph is already translated
            if (
                id(last_curr_paragraph) in translated_ids
                or id(first_next_paragraph) in translated_ids
            ):
                continue

            # Build font maps for both pages
            curr_font_map, curr_xobj_font_map = self._build_font_maps(page_curr)
            next_font_map, next_xobj_font_map = self._build_font_maps(page_next)

            # Merge font maps
            merged_font_map = {**curr_font_map, **next_font_map}
            merged_xobj_font_map = {**curr_xobj_font_map, **next_xobj_font_map}

            # Calculate total token count
            total_token_count = self.calc_token_count(
                last_curr_paragraph.unicode
            ) + self.calc_token_count(first_next_paragraph.unicode)

            # Use LLM translator for cross-page paragraphs
            cross_page_paragraphs = [last_curr_paragraph, first_next_paragraph]

            self.mid += 1
            executor.submit(
                self.il_translator.translate_paragraph,
                last_curr_paragraph,
                pbar,
                tracker.new_cross_page().new_paragraph(),
                merged_font_map,
                merged_xobj_font_map,
                priority=1048576 - total_token_count,
                paragraph_token_count=self.calc_token_count(
                    last_curr_paragraph.unicode
                ),
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

            executor.submit(
                self.il_translator.translate_paragraph,
                first_next_paragraph,
                pbar,
                tracker.new_cross_page().new_paragraph(),
                merged_font_map,
                merged_xobj_font_map,
                priority=1048576 - total_token_count,
                paragraph_token_count=self.calc_token_count(
                    first_next_paragraph.unicode
                ),
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

            # Mark paragraphs as translated
            translated_ids.add(id(last_curr_paragraph))
            translated_ids.add(id(first_next_paragraph))

    def process_cross_column_paragraph(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: DocumentTranslateTracker | None = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set[int] | None = None,
    ):
        """Process cross-column paragraphs using LLM (Qwen-Plus)"""
        self.translation_config.raise_if_cancelled()

        if tracker is None:
            tracker = DocumentTranslateTracker()
        if translated_ids is None:
            translated_ids = set()

        # Filter body-text paragraphs maintaining original order
        body_paragraphs = self._filter_paragraphs(
            page, translated_ids, require_body_text=True
        )
        if len(body_paragraphs) < 2:
            return

        # Build font maps once for the whole page
        page_font_map, page_xobj_font_map = self._build_font_maps(page)

        for idx in range(len(body_paragraphs) - 1):
            p1 = body_paragraphs[idx]
            p2 = body_paragraphs[idx + 1]

            # Skip already translated
            if id(p1) in translated_ids or id(p2) in translated_ids:
                continue

            # Safety checks for box information
            if not (
                p1.box and p2.box and p1.box.y2 is not None and p2.box.y2 is not None
            ):
                continue

            if p2.box.y2 - p1.box.y2 <= 20:
                continue

            # Use LLM translator for cross-column paragraphs
            self.mid += 1
            executor.submit(
                self.il_translator.translate_paragraph,
                p1,
                pbar,
                tracker.new_cross_column().new_paragraph(),
                page_font_map,
                page_xobj_font_map,
                priority=1048576 - self.calc_token_count(p1.unicode),
                paragraph_token_count=self.calc_token_count(p1.unicode),
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

            executor.submit(
                self.il_translator.translate_paragraph,
                p2,
                pbar,
                tracker.new_cross_column().new_paragraph(),
                page_font_map,
                page_xobj_font_map,
                priority=1048576 - self.calc_token_count(p2.unicode),
                paragraph_token_count=self.calc_token_count(p2.unicode),
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

            translated_ids.add(id(p1))
            translated_ids.add(id(p2))

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
        executor2: PriorityThreadPoolExecutor | None = None,
        translated_ids: set | None = None,
    ):
        """Process regular paragraphs using QwenMT"""
        self.translation_config.raise_if_cancelled()
        page_font_map = {}
        for font in page.pdf_font:
            page_font_map[font.font_id] = font
        page_xobj_font_map = {}
        for xobj in page.pdf_xobject:
            page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
            for font in xobj.pdf_font:
                page_xobj_font_map[xobj.xobj_id][font.font_id] = font

        paragraphs = []
        total_token_count = 0

        for paragraph in page.pdf_paragraph:
            # Check if already translated
            if id(paragraph) in translated_ids:
                continue

            # Check basic validation
            if paragraph.debug_id is None or paragraph.unicode is None:
                continue

            # Check CID paragraph - advance progress bar if filtered out
            if is_cid_paragraph(paragraph):
                if pbar:
                    pbar.advance(1)
                continue

            # Check minimum length - advance progress bar if filtered out
            if len(paragraph.unicode) < self.translation_config.min_text_length:
                if pbar:
                    pbar.advance(1)
                continue

            if is_pure_numeric_paragraph(paragraph):
                if pbar:
                    pbar.advance(1)
                continue

            if is_placeholder_only_paragraph(paragraph):
                if pbar:
                    pbar.advance(1)
                continue

            # Use QwenMT for regular paragraphs
            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            translated_ids.add(id(paragraph))

            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = (
                    copy.deepcopy(paragraph)
                )

            # Process in batches for QwenMT
            if total_token_count > 200 or len(paragraphs) > 5:
                self.mid += 1
                executor.submit(
                    self.translate_paragraph_qwenmt,
                    BatchParagraph(paragraphs, tracker),
                    pbar,
                    page_font_map,
                    page_xobj_font_map,
                    self.translation_config.shared_context_cross_split_part.first_paragraph,
                    self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                    executor2,
                    priority=1048576 - total_token_count,
                    paragraph_token_count=total_token_count,
                    mp_id=self.mid,
                )
                paragraphs = []
                total_token_count = 0

        if paragraphs:
            self.mid += 1
            executor.submit(
                self.translate_paragraph_qwenmt,
                BatchParagraph(paragraphs, tracker),
                pbar,
                page_font_map,
                page_xobj_font_map,
                self.translation_config.shared_context_cross_split_part.first_paragraph,
                self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                executor2,
                priority=1048576 - total_token_count,
                paragraph_token_count=total_token_count,
                mp_id=self.mid,
            )

    def translate_paragraph_qwenmt(
        self,
        batch_paragraph: BatchParagraph,
        pbar: tqdm | None = None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        executor: PriorityThreadPoolExecutor | None = None,
        paragraph_token_count: int = 0,
        mp_id: int = 0,
    ):
        """Translate paragraphs using QwenMT for regular translation"""
        self.translation_config.raise_if_cancelled()
        should_translate_paragraph = []

        try:
            inputs = []
            paragraph_unicodes = []

            for i in range(len(batch_paragraph.paragraphs)):
                paragraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]

                # Pre-process paragraph
                text, translate_input = self.il_translator.pre_translate_paragraph(
                    paragraph, tracker, page_font_map, xobj_font_map
                )
                if text is None:
                    if pbar:
                        pbar.advance(1)
                    continue

                tracker.record_multi_paragraph_id(mp_id)
                should_translate_paragraph.append(i)
                inputs.append((text, translate_input, paragraph, tracker))
                paragraph_unicodes.append(paragraph.unicode)

            if not inputs:
                return

            # Process paragraphs individually with QwenMT
            for i, (text, translate_input, paragraph, tracker) in enumerate(inputs):
                should_fallback = True
                try:
                    # Use QwenMT for translation
                    translated_text = self.translate_engine.do_translate(
                        text,
                        rate_limit_params={
                            "paragraph_token_count": paragraph_token_count
                        },
                    )

                    # Basic quality checks
                    input_token_count = self.calc_token_count(text)
                    output_token_count = self.calc_token_count(translated_text)

                    # Clean up any excessive punctuation
                    translated_text = re.sub(r"[. 。…，]{20,}", ".", translated_text)

                    # Check if translation is reasonable
                    if (
                        text.strip() == translated_text.strip()
                        and input_token_count > 10
                    ):
                        logger.warning(
                            "QwenMT translation result is the same as input, fallback."
                        )
                        continue

                    if not (0.3 < output_token_count / max(input_token_count, 1) < 3):
                        logger.warning(
                            f"QwenMT translation result length unusual. Input: {input_token_count}, Output: {output_token_count}"
                        )
                        continue

                    # Apply the translation to the paragraph
                    self.il_translator.post_translate_paragraph(
                        paragraph, tracker, translate_input, translated_text
                    )
                    should_fallback = False
                    self.ok_count += 1
                    if pbar:
                        pbar.advance(1)

                except Exception as e:
                    error_message = f"Error in QwenMT translation: {e}"
                    logger.warning(error_message)

                finally:
                    self.total_count += 1
                    if should_fallback:
                        self.fallback_count += 1
                        logger.warning(
                            f"Fallback to LLM translation for paragraph id: {paragraph.debug_id}"
                        )
                        # Restore original text for fallback
                        paragraph.unicode = paragraph_unicodes[i]
                        # Use LLM translator as fallback
                        paragraph_token_count_single = self.calc_token_count(
                            paragraph.unicode
                        )
                        executor.submit(
                            self.il_translator.translate_paragraph,
                            paragraph,
                            pbar,
                            tracker,
                            page_font_map,
                            xobj_font_map,
                            priority=1048576 - paragraph_token_count_single,
                            paragraph_token_count=paragraph_token_count_single,
                            title_paragraph=title_paragraph,
                            local_title_paragraph=local_title_paragraph,
                        )

        except Exception as e:
            error_message = (
                f"Error {e} during QwenMT batch translation, fallback to LLM"
            )
            logger.warning(error_message)
            # Fallback all paragraphs to LLM
            if not should_translate_paragraph:
                should_translate_paragraph = list(
                    range(len(batch_paragraph.paragraphs))
                )

            for i in should_translate_paragraph:
                paragraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]
                if paragraph.debug_id is None:
                    continue
                paragraph_token_count_single = self.calc_token_count(paragraph.unicode)
                executor.submit(
                    self.il_translator.translate_paragraph,
                    paragraph,
                    pbar,
                    tracker,
                    page_font_map,
                    xobj_font_map,
                    priority=1048576 - paragraph_token_count_single,
                    paragraph_token_count=paragraph_token_count_single,
                    title_paragraph=title_paragraph,
                    local_title_paragraph=local_title_paragraph,
                )
