import re
from typing import Union, Literal

from yadt.document_il import (
    Box,
    Page,
    PdfCharacter,
    PdfLine,
    PdfParagraph,
    PdfParagraphComposition,
)
from yadt.document_il.utils.layout_helper import (
    Layout,
    add_space_dummy_chars,
    get_char_unicode_string,
    HEIGHT_NOT_USFUL_CHAR_IN_CHAR,
)


class ParagraphFinder:
    def update_paragraph_data(self, paragraph: PdfParagraph, update_unicode=False):
        if not paragraph.pdf_paragraph_composition:
            return

        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                chars.extend(composition.pdf_formula.pdf_character)
            else:
                raise Exception(
                    "Unexpected composition type"
                    " in PdfParagraphComposition. "
                    "This type only appears in the IL "
                    "after the translation is completed."
                )

        if update_unicode:
            paragraph.unicode = get_char_unicode_string(chars)
        # 更新边界框
        min_x = min(char.box.x for char in chars)
        min_y = min(char.box.y for char in chars)
        max_x = max(char.box.x2 for char in chars)
        max_y = max(char.box.y2 for char in chars)
        paragraph.box = Box(min_x, min_y, max_x, max_y)
        paragraph.vertical = chars[0].vertical

        paragraph.first_line_indent = False
        if (
            paragraph.pdf_paragraph_composition[0].pdf_line
            and paragraph.pdf_paragraph_composition[0].pdf_line.pdf_character[0].box.x
            - paragraph.box.x
            > 1
        ):
            paragraph.first_line_indent = True

    def update_line_data(self, line: PdfLine):
        min_x = min(char.box.x for char in line.pdf_character)
        min_y = min(char.box.y for char in line.pdf_character)
        max_x = max(char.box.x2 for char in line.pdf_character)
        max_y = max(char.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    def process(self, document):
        for page in document.page:
            self.process_page(page)

    def process_page(self, page: Page):
        # 第一步：根据 layout 创建 paragraphs
        # 在这一步中，page.pdf_character 中的字符会被移除
        paragraphs = self.create_paragraphs(page)
        page.pdf_paragraph = paragraphs

        # 第二步：处理段落中的空格和换行符
        for paragraph in paragraphs:
            add_space_dummy_chars(paragraph)
            self.process_paragraph_spacing(paragraph)
            self.update_paragraph_data(paragraph)

        # 第三步：计算所有行宽度的中位数
        median_width = self.calculate_median_line_width(paragraphs)

        # 第四步：处理独立段落
        self.process_independent_paragraphs(paragraphs, median_width)

        for paragraph in paragraphs:
            self.update_paragraph_data(paragraph, update_unicode=True)

    def create_paragraphs(self, page: Page) -> list[PdfParagraph]:
        paragraphs: list[PdfParagraph] = []
        current_paragraph: PdfParagraph | None = None
        current_layout: Layout | None = None
        current_line_chars: list[PdfCharacter] = []
        skip_chars = []

        for char in page.pdf_character:
            char_layout = self.get_layout(char, page)
            if not self.is_text_layout(char_layout):
                skip_chars.append(char)
                continue

            # 检查是否需要开始新行
            if current_line_chars and Layout.is_newline(current_line_chars[-1], char):
                # 创建新行
                if current_line_chars:
                    line = self.create_line(current_line_chars)
                    if current_paragraph is None:
                        current_paragraph = PdfParagraph(
                            pdf_paragraph_composition=[line],
                        )
                        paragraphs.append(current_paragraph)
                    else:
                        current_paragraph.pdf_paragraph_composition.append(line)
                        self.update_paragraph_data(current_paragraph)
                current_line_chars = []

            # 检查是否需要开始新段落
            if current_layout is None or char_layout.id != current_layout.id:
                if current_line_chars:
                    line = self.create_line(current_line_chars)
                    if current_paragraph is not None:
                        current_paragraph.pdf_paragraph_composition.append(line)
                        self.update_paragraph_data(current_paragraph)
                    else:
                        current_paragraph = PdfParagraph(
                            pdf_paragraph_composition=[line],
                        )
                        self.update_paragraph_data(current_paragraph)
                        paragraphs.append(current_paragraph)
                    current_line_chars = []
                current_paragraph = None
                current_layout = char_layout

            current_line_chars.append(char)

        # 处理最后一行的字符
        if current_line_chars:
            line = self.create_line(current_line_chars)
            if current_paragraph is None:
                current_paragraph = PdfParagraph(
                    pdf_paragraph_composition=[line],
                )
                paragraphs.append(current_paragraph)
            else:
                current_paragraph.pdf_paragraph_composition.append(line)
                self.update_paragraph_data(current_paragraph)

        page.pdf_character = skip_chars

        return paragraphs

    def process_paragraph_spacing(self, paragraph: PdfParagraph):
        if not paragraph.pdf_paragraph_composition:
            return

        # 处理行级别的空格
        processed_lines = []
        for composition in paragraph.pdf_paragraph_composition:
            if not composition.pdf_line:
                continue

            line = composition.pdf_line
            if not "".join(
                (x.char_unicode for x in line.pdf_character)
            ).strip():  # 跳过完全空白的行
                continue

            # 处理行内字符的尾随空格
            processed_chars = []
            for char in line.pdf_character:
                if not char.char_unicode.isspace():
                    processed_chars = processed_chars + [char]
                elif processed_chars:  # 只有在有非空格字符后才考虑保留空格
                    processed_chars.append(char)

            # 移除尾随空格
            while processed_chars and processed_chars[-1].char_unicode.isspace():
                processed_chars.pop()

            if processed_chars:  # 如果行内还有字符
                line = self.create_line(processed_chars)
                processed_lines.append(line)

        paragraph.pdf_paragraph_composition = processed_lines
        self.update_paragraph_data(paragraph)

    def is_text_layout(self, layout: Layout):
        return layout is not None and layout.name in [
            "plain text",
            "title",
            "abandon",
            "figure_caption",
            "table_caption",
        ]

    def get_layout(
        self,
        char: PdfCharacter,
        page: Page,
        xy_mode: Union[
            Literal["topleft"], Literal["bottomright"], Literal["middle"]
        ] = "middle",
    ):
        # 这几个符号，解析出来的大小经常只有实际大小的一点点。
        if xy_mode != "bottomright" and char.char_unicode in HEIGHT_NOT_USFUL_CHAR_IN_CHAR:
            return self.get_layout(char, page, "bottomright")
        # current layouts
        # {
        #     "title",
        #     "plain text",
        #     "abandon",
        #     "figure",
        #     "figure_caption",
        #     "table",
        #     "table_caption",
        #     "table_footnote",
        #     "isolate_formula",
        #     "formula_caption",
        # }
        layout_priority = [
            "formula_caption",
            "isolate_formula",
            "table_footnote",
            "table",
            "figure",
            "table_caption",
            "figure_caption",
            "abandon",
            "plain text",
            "title",
        ]
        char_box = char.box
        if xy_mode == "topleft":
            char_x = char_box.x
            char_y = char_box.y2
        elif xy_mode == "bottomright":
            char_x = char_box.x2
            char_y = char_box.y
        elif xy_mode == "middle":
            char_x = (char_box.x + char_box.x2) / 2
            char_y = (char_box.y + char_box.y2) / 2
        else:
            raise ValueError(f"Invalid xy_mode: {xy_mode}")
        # 按照优先级顺序检查每种布局
        matching_layouts = {}
        for layout in page.page_layout:
            layout_box = layout.box
            if (
                layout_box.x <= char_x <= layout_box.x2
                and layout_box.y <= char_y <= layout_box.y2
            ):
                matching_layouts[layout.class_name] = Layout(
                    layout.id, layout.class_name
                )

        # 按照优先级返回最高优先级的布局
        for layout_name in layout_priority:
            if layout_name in matching_layouts:
                return matching_layouts[layout_name]

        if xy_mode == "middle":
            return self.get_layout(char, page, "topleft")
        if xy_mode == "topleft":
            return self.get_layout(char, page, "bottomright")
        return None

    def create_line(self, chars: list[PdfCharacter]) -> PdfParagraphComposition:
        assert chars

        line = PdfLine(pdf_character=chars)
        self.update_line_data(line)
        return PdfParagraphComposition(pdf_line=line)

    def calculate_median_line_width(self, paragraphs: list[PdfParagraph]) -> float:
        # 收集所有行的宽度
        line_widths = []
        for paragraph in paragraphs:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line_widths.append(line.box.x2 - line.box.x)

        if not line_widths:
            return 0.0

        # 计算中位数
        line_widths.sort()
        mid = len(line_widths) // 2
        if len(line_widths) % 2 == 0:
            return (line_widths[mid - 1] + line_widths[mid]) / 2
        return line_widths[mid]

    def process_independent_paragraphs(
        self, paragraphs: list[PdfParagraph], median_width: float
    ):
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            if len(paragraph.pdf_paragraph_composition) <= 1:  # 跳过只有一行的段落
                i += 1
                continue

            j = 1
            while j < len(paragraph.pdf_paragraph_composition):
                prev_composition = paragraph.pdf_paragraph_composition[j - 1]
                if not prev_composition.pdf_line:
                    j += 1
                    continue

                prev_line = prev_composition.pdf_line
                prev_width = prev_line.box.x2 - prev_line.box.x
                prev_text = "".join([c.char_unicode for c in prev_line.pdf_character])

                # 检查是否包含连续的点（至少 20 个）
                # 如果有至少连续 20 个点，则代表这是目录条目
                if re.search(r"\.{20,}", prev_text):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break

                # 如果前一行宽度小于中位数的一半，将当前行及后续行分割成新段落
                # if prev_width < median_width * 0.8:
                #     # 创建新的段落
                #     new_paragraph = PdfParagraph(
                #         box=Box(0, 0, 0, 0),  # 临时边界框
                #         pdf_paragraph_composition=(
                #             paragraph.pdf_paragraph_composition[j:]
                #         ),
                #         unicode="",
                #     )
                #     # 更新原段落
                #     paragraph.pdf_paragraph_composition = (
                #         paragraph.pdf_paragraph_composition[:j]
                #     )
                #
                #     # 更新两个段落的数据
                #     self.update_paragraph_data(paragraph)
                #     self.update_paragraph_data(new_paragraph)
                #
                #     # 在原段落后插入新段落
                #     paragraphs.insert(i + 1, new_paragraph)
                #     break
                j += 1
            i += 1
