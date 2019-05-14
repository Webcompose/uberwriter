import json
import operator
import re
import time
import warnings
from dataclasses import dataclass

from uberwriter import helpers


@dataclass
class Result:
    """Type of the result from Pandoc's type list."""
    type: str
    """Start of the result in the text."""
    start: int
    """End of the result in the text."""
    end: int
    """Extra metadata, such as:
     * content_start / content_end: start/end of "bold" in **bold**, "link" in "[link](url)", etc
     * level: a header's level
     * header_start / header_end / rows_start / rows_end / caption_start / caption_end: tables
    """
    extras: dict


class PandocSourceMapper:
    """PandocSourceMapper walks Pandoc's AST while iterating the original text itself, adding extra
    metadata to each node such as its location in the original text, and other relevant data related
    with each individual type of node (eg. where a table's header/caption/rows start and end, where
    a links' url/title starts and ends, or which levela  given header is).

    The typical approach of using regular expressions is much faster, but also much less accurate.
    Provided it doesn't break, walking Pandoc's AST effectively provides a fully accurate source
    map, absent of edge cases or unsupported scenarios. Everything that Pandoc understands can be
    mapped onto the original text, including secondary metadata.

    The workflow implemented here is directly dependant on Pandoc 2.X's AST implementation.
    AST changes are signalled by major versions, meaning this will likely break once 3.X lands.

    There are specific details to the parsing of each type that are handled and documented below.

    Relevant documentation/discussions:
    - https://github.com/jgm/pandoc-types/blob/master/Text/Pandoc/Definition.hs
    - https://github.com/jgm/pandoc/issues/4565
    - https://pandoc.org/MANUAL.html
    """

    code_block_delim = re.compile(r"^\s{0,3}[`~]{3,}")

    header_setext_delim = re.compile(r"^\s{0,3}(?:-+|=+)\s*$")

    table_delim = re.compile(r"^\s{0,3}(?:[-\s]+|[-+:]+)\s*$")

    smallcaps_delims = [
        ("[", "]{.smallcaps}"),
        ("<span ", "</span>")
    ]

    citation_delims = ("[", "]")

    math_delims = {
        "DisplayMath": "$$",
        "InlineMath": "$"
    }

    footnote_identifier = re.compile(r"\[\^[^\s]+\]")

    # https://github.com/jgm/pandoc/blob/f3080c0c22470e7ecccbef86c1cd0b1339a6de9b/src/Text/Pandoc/Readers/Markdown.hs#L1527-L1547
    # https://github.com/jgm/pandoc/blob/f3080c0c22470e7ecccbef86c1cd0b1339a6de9b/src/Text/Pandoc/XML.hs#L33-L38
    # http://rabbit.eng.miami.edu/info/htmlchars.html
    escape_table = {
        "\\\\": "\\",
        "\\`": "`",
        "\\*": "*",
        "\\_": "_",
        "\\{": "{",
        "\\}": "}",
        "\\[": "[",
        "\\]": "]",
        "\\(": "(",
        "\\)": ")",
        "\\#": "#",
        "\\+": "+",
        "\\-": "-",
        "\\.": ".",
        "\\!": "!",
        "\\~": "~",
        "\\\"": "\"",
        "\\<": "<",
        "\\>": ">",
        "\\\n": "\n",
        " ": "\xA0",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&nbsp;": " ",
        "&iexcl;": "¡",
        "&cent;": "¢",
        "&pound;": "£",
        "&curren;": "¤",
        "&yen;": "¥",
        "&brvbar;": "¦",
        "&sect;": "§",
        "&uml;": "¨",
        "&copy;": "©",
        "&ordf;": "ª",
        "&laquo;": "«",
        "&not;": "¬",
        "&shy;": "­",
        "&reg;": "®",
        "&macr;": "¯",
        "&deg;": "°",
        "&plusmn;": "±",
        "&sup2;": "²",
        "&sup3;": "³",
        "&acute;": "´",
        "&micro;": "µ",
        "&para;": "¶",
        "&middot;": "·",
        "&cedil;": "¸",
        "&sup1;": "¹",
        "&ordm;": "º",
        "&raquo;": "»",
        "&frac14;": "¼",
        "&frac12;": "½",
        "&frac34;": "¾",
        "&iquest;": "¿",
        "&times;": "×",
        "&divide;": "÷",
        "&ETH;": "Ð",
        "&eth;": "ð",
        "&THORN;": "Þ",
        "&thorn;": "þ",
        "&AElig;": "Æ",
        "&aelig;": "æ",
        "&OElig;": "Œ",
        "&oelig;": "œ",
        "&Aring;": "Å",
        "&Oslash;": "Ø",
        "&Ccedil;": "Ç",
        "&ccedil;": "ç",
        "&szlig;": "ß",
        "&Ntilde;": "Ñ",
        "&ntilde;": "ñ"
    }
    escape_table_keys = tuple(escape_table.keys())
    escape_table_values = tuple(escape_table.values())

    def __init__(self, text):
        self.text = text

        self.results = []

        # Blocks
        # https://github.com/jgm/pandoc-types/blob/master/Text/Pandoc/Definition.hs#L219-L242
        self.block_types = {
            "Plain": (self.walk_multiple, lambda args: [args]),
            "Para": (self.walk_multiple, lambda args: [args]),
            "LineBlock": (self.walk_multiple_nested, lambda args: [args]),
            "CodeBlock": (self.walk_code_block, lambda args: args),
            "RawBlock": (self.walk_str, lambda args: [args[1]]),
            "BlockQuote": (self.walk_multiple, lambda args: [args]),
            "OrderedList": (self.walk_list, lambda args: [args[1]]),
            "BulletList": (self.walk_list, lambda args: [args]),
            "DefinitionList": (self.walk_multiple_to_multiple_nested, lambda arg: [arg]),
            "Header": (self.walk_header, lambda args: args),
            "HorizontalRule": (self.walk_horizontal_rule, lambda args: []),
            "Table": (self.walk_table, lambda args: args),
            "Div": (self.walk_div, lambda args: args),
            "Null": (self.walk_nothing, lambda args: []),
        }
        # Inlines
        # https://github.com/jgm/pandoc-types/blob/master/Text/Pandoc/Definition.hs#L255-L274
        self.inline_types = {
            "Str": (self.walk_str, lambda arg: [arg]),
            "Emph": (self.walk_multiple, lambda args: [args, 1]),
            "Strong": (self.walk_multiple, lambda args: [args, 2]),
            "Strikeout": (self.walk_multiple, lambda args: [args, 2]),
            "Superscript": (self.walk_multiple, lambda args: [args, 1]),
            "Subscript": (self.walk_multiple, lambda args: [args, 1]),
            "SmallCaps": (self.walk_small_caps, lambda args: [args]),
            "Quoted": (self.walk_multiple, lambda args: [args[1], 1]),
            "Cite": (self.walk_cite, lambda args: args),
            "Code": (self.walk_code, lambda args: [args[1]]),
            "Space": (self.walk_spaces, lambda args: []),
            "SoftBreak": (self.walk_spaces, lambda args: []),
            "LineBreak": (self.walk_spaces, lambda args: []),
            "Math": (self.walk_math, lambda args: args),
            "RawInline": (self.walk_str, lambda args: [args[1]]),
            "Link": (self.walk_link, lambda args: args),
            "Image": (self.walk_link, lambda args: args),
            "Note": (self.walk_note, lambda args: [args]),
            "Span": (self.walk_multiple, lambda args: [args[1]]),
        }
        self.types = dict(self.block_types, **self.inline_types)

        # Types where leading and trailing spaces are not expected.
        # Includes all inlines except Space, SoftBreak and LineBreak.
        self.non_spaced_types = ["Str", "Emph", "Strong", "Strikeout", "Superscript", "Subscript",
                                 "SmallCaps", "Quoted", "Cite", "Code", "Math", "RawInline", "Link",
                                 "Image", "Note", "Span"]

        # Types that span whole lines.
        # Includes all blocks except Plain, RawBlock and Null.
        self.paragraph_types = ["Para", "LineBlock", "CodeBlock", "BlockQuote",
                                "OrderedList", "BulletList", "DefinitionList", "Header",
                                "HorizontalRule", "Table", "Div"]

    def walk(self):
        s1 = time.time()
        converted = helpers.pandoc_convert(self.text, to="json", format_ext="-smart")
        s2 = time.time()
        blocks = json.loads(converted)["blocks"]
        self.walk_multiple(0, blocks)
        s3 = time.time()
        print("Time to parse: {}s\nTime to walk: {}s\nTotal: {}"s.format(s2 - s1, s3 - s2, s3 - s1))
        return self.results

    # AST parsers
    def walk_multiple_to_multiple_nested(self, index, entries_to_entries_nested):
        """[([type], [[type]])]"""

        start = end = index
        for i, entry_to_entry_nested in enumerate(entries_to_entries_nested):
            entry, entries_nested = entry_to_entry_nested
            s, end, _ = self.walk_multiple(end, entry)
            if i == 0:
                start = s
            _, end, _ = self.walk_multiple_nested(end, entries_nested)
        return start, end, {}

    def walk_multiple_nested(self, index, entries_nested, nesting=1):
        """[[type]] for nesting = 1, [[[type]]] for nesting = 2, etc"""

        nesting -= 1
        start = end = index
        for i, entries in enumerate(entries_nested):
            if nesting > 0:
                s, end, _ = self.walk_multiple_nested(end, entries, nesting)
            else:
                s, end, _ = self.walk_multiple(end, entries)
            if i == 0:
                start = s
        return start, end, {}

    def walk_multiple(self, index, entries, padding=0):
        """[type]"""

        start = end = index
        results = []
        for i, entry in enumerate(entries):
            result = self.walk_one(end, entry)
            if result:
                results.append(result)
                # Start/end can only be inherited for ordered types, as unordered types
                # can be anywhere in the text.
                if self.__is_last_result_ordered(results):
                    if i == 0:
                        start = result.start
                    end = result.end
        extras = {"content_start": start, "content_end": end} if padding != 0 else {}
        return start - padding, end + padding, extras

    def walk_one(self, index, entry):
        t = entry["t"]
        if t not in self.types:
            warnings.warn("Unknown Pandoc type '{}', please report this bug".format(t))
            return None

        # Walk using function and args function for the current type.
        fn, args_fn = self.types[t]
        result = Result(t, *fn(index, *args_fn(entry.get("c"))))
        if result.start < 0 or result.end < 0 or result.start >= result.end:
            warnings.warn("Invalid result '{}', please report this bug".format(result))
            return None

        # Remove extraneous spaces from non-space types.
        if result.type in self.non_spaced_types:
            while self.text[result.start].isspace():
                result.start += 1
            while self.text[result.end - 1].isspace():
                result.end -= 1

        # Expand start/end of paragraph types to include the whole line.
        if result.type in self.paragraph_types:
            result.extras.update({"s": result.start, "e": result.end})
            result.start = self.__get_line(result.start)[0]
            result.end = self.__get_line(result.end)[1]

        self.results.append(result)
        return result

    # Block walkers.
    # Upon saving, each block's start/end will be adjusted to include the whole line.
    # Walking doesn't need to ensure the whole line is included, just the content itself.

    def walk_code_block(self, index, _attr, string):
        content_start = content_end = index

        # Spacing can be troublesome for code blocks, as the output won't match the source:
        # * Tabs can get converted to spaces
        # * Spaces in the beginning of each line won't match due to indentation
        for i, line in enumerate(string.split()):
            s, content_end = self.__find(line, content_end)
            if i == 0:
                content_start = s
        start = content_start
        end = content_end

        # Include the previous / next line if they are fences.
        line_before_start, line_before_end = self.__get_line_before(content_start)
        line_after_start, line_after_end = self.__get_line_after(content_end)
        if self.code_block_delim.match(self.text[line_before_start:line_before_end]) and \
                self.code_block_delim.match(self.text[line_after_start:line_after_end]):
            start = line_before_start
            end = line_after_end

        return start, end, {"content_start": content_start, "content_end": content_end}

    def walk_list(self, index, nested_entries):
        start, end, _ = self.walk_multiple_nested(index, nested_entries)
        # Upon saving, start/end will include their whole line and differ from content start/end.
        return start, end, {"content_start": start, "content_end": end}

    def walk_header(self, index, level, _attr, entries):
        start, end, _ = self.walk_multiple(index, entries)
        line_after_start, line_after_end = self.__get_line_after(end)
        if self.header_setext_delim.match(self.text[line_after_start:line_after_end]):
            end = line_after_end
        # Upon saving, start/end will include their whole line and differ from content start/end.
        return start, end, {"level": level, "content_start": start, "content_end": end}

    def walk_horizontal_rule(self, index):
        while self.text[index].isspace():
            index += 1
        start, end = self.__get_line(index)
        while self.text[start].isspace():
            start += 1
        while self.text[end - 1].isspace():
            end -= 1
        # Upon saving, start/end will include the whole line and differ from content start/end.
        return start, end, {"content_start": start, "content_end": end}

    def walk_table(self, index, caption, _alignment, _column_widths, header, rows):
        start = index
        # Locate header, rows and caption.
        header_start, header_end, _ = self.walk_multiple_nested(start, header)
        rows_start, rows_end, _ = self.walk_multiple_nested(header_end, rows, nesting=2)
        caption_start, caption_end, _ = self.walk_multiple(start, caption)
        # Figure out what comes first and last, considering the caption can appear before or after.
        if header_start != header_end and caption_start != caption_end:
            start = min(header_start, caption_start)
        elif header_start != header_end:
            start = header_start
        elif caption_start != caption_end:
            start = min(caption_start, rows_start)
        else:
            start = rows_start
        if caption_start != caption_end:
            end = max(caption_end, rows_end)
        else:
            end = rows_end
        # Offset start/end to include top/bottom delimiters if present.
        line_before_start, line_before_end = self.__get_line_before(start)
        if self.table_delim.match(self.text[line_before_start:line_before_end]):
            start = line_before_start
        line_after_start, line_after_end = self.__get_line_after(end)
        if self.table_delim.match(self.text[line_after_start:line_after_end]):
            end = line_after_end
        # Add extras.
        extras = {"rows_start": rows_start, "rows_end": rows_end}
        if header_start != header_end:
            extras.update({"header_start": header_start, "header_end": header_end})
        if caption_start != caption_end:
            extras.update({"caption_start": caption_start, "caption_end": caption_end})
        return start, end, extras

    def walk_div(self, index, attr, entries):
        start, end, _ = self.walk_multiple(index, entries)
        identifier, _, _ = attr
        extras = {"identifier": identifier} if identifier else {}
        return start, end, extras

    # Inline walkers.
    # Upon saving, non-space inline types will be trimmed. Walking doesn't need to trim them.
    def walk_str(self, index, string):
        if self.text.startswith(string, index):
            start = index
            end = start + len(string)
        else:
            start, end = self.__find_unescaped(string, index)
        return start, end, {}

    def walk_small_caps(self, index, entries):
        start, end, _ = self.walk_multiple(index, entries)
        content_start, content_end = start, end
        for smallcaps_start, smallcaps_end in self.smallcaps_delims:
            if self.text.startswith(smallcaps_end, end):
                start = self.text.rfind(smallcaps_start, 0, end)
                end += len(smallcaps_end)
                break
        return start, end, {"content_start": content_start, "content_end": content_end}

    def walk_cite(self, index, citations, _bibliography_entries):
        start = end = citation_start = index
        capture_brackets = True
        extras_citations = []
        # Walk each citation, adding it to the extras map.
        for i, citation in enumerate(citations):
            citation_start, citation_end = self.__find("@" + citation["citationId"], citation_start)
            if i == 0:
                start = citation_start
            end = citation_end
            capture_brackets &= citation["citationMode"]["t"] != "AuthorInText"
            extras_citations.append({"start": citation_start, "end": citation_end})
        # Include end delimiter except if any is "AuthorInText", in which case it's not present.
        if start != end and capture_brackets:
            citation_start_delim, citation_end_delim = self.citation_delims
            while start >= 0 and self.text[start] != citation_start_delim:
                start -= 1
            while self.text[end - 1] != citation_end_delim:
                end += 1
        return start, end, {"citations": extras_citations}

    def walk_code(self, index, string):
        start = end = index
        # Search word by word, as inline code can have line breaks which won't exist in the output.
        for i, line in enumerate(string.split()):
            s, end = self.__find(line, end)
            if i == 0:
                start = s
        # Pad start/end to include start/end fences.
        return start - 1, end + 1, {"content_start": start, "content_end": end}

    def walk_spaces(self, index):
        start = index
        while not self.text[start].isspace():
            start += 1
        end = start
        while self.text[end].isspace():
            end += 1
        return start, end, {}

    def walk_math(self, index, math_type, string):
        start, end = self.__find(string, index)
        padding = len(self.math_delims[math_type["t"]])
        return start - padding, end + padding, {"content_start": start, "content_end": end}

    def walk_link(self, index, _attr, entries, target):
        start = index
        # Locate content.
        content_start, content_end, _ = self.walk_multiple(start, entries)
        # Locate url and title, if present.
        url_start = url_end = title_start = title_end = end = content_end
        url, title = target
        if url:
            url_start, url_end = self.__find(url, end)
        if title:
            title_start, title_end = self.__find(title, end)
        end = max(url_end, title_end)
        # Ensure url/title are on the same line as the link.
        # Reference links' metadata is unordered.
        if self.__get_line(content_end)[0] != self.__get_line(end)[0]:
            end = content_end
        elif url_start != url_end:
            end += 1  # Include closing parens when URL is found and is in the same line.
        # Add extras.
        extras = {"content_start": content_start, "content_end": content_end}
        if url_start != url_end:
            extras.update({"url_start": url_start, "url_end": url_end})
        if title_start != title_end:
            extras.update({"title_start": title_start, "title_end": title_end})
        return start, end, {"content_start": content_start, "content_end": content_end}

    def walk_note(self, index, entries):
        start = end = index
        if self.text[start:].startswith("^["):
            # Inline note.
            start, end, _ = self.walk_multiple(start, entries)
            extras = {"type": "inline", "content_start": start, "content_end": end}
            end += 1  # Include closing bracket.
        else:
            # Footnotes.
            extras = {"type": "footnote"}
            match = self.footnote_identifier.match(self.text[start:])
            if match:
                match = re.search(
                    "^\\s{{0,3}}{}: ".format(re.escape(match.group())), self.text, re.MULTILINE)
            if match:
                start, end, _ = self.walk_multiple(match.start(), entries)
                extras.update({"content_start": start, "content_end": end})
        return start, end, extras

    @staticmethod
    def walk_nothing(index):
        return index, index, {}

    # Utilities

    def __find(self, sub, start=None, end=None):
        start = end = self.text.find(sub, start, end)
        if start >= 0:
            end = start + len(sub)
        return start, end

    def __find_unescaped(self, sub, start=None, end=None):
        """Finds `sub` in `string`, starting at `start` and ending at `end`.

        Functionally, this is equivalent to `str.find`, but escaped characters are
        matched to their unescaped equivalent, and carriege returns ignored."""

        if start is None:
            start = 0
        if end is None:
            end = len(self.text)
        if start > end or start < 0 or end > len(self.text):
            return -1, -1
        if not sub:
            return start, start

        index = 0
        start_match = -1
        i = start
        while i < end:
            increment = -1
            if self.text[i] == sub[index]:
                increment = 1
            elif start_match != -1 and self.text[i] == "\r":
                increment = 0
            elif sub[index] in self.escape_table_values:
                for c in self.escape_table:
                    if self.text[i:].startswith(c):
                        increment = len(c)
                        break

            # noinspection PyChainedComparisons
            if increment > 0:
                index += 1
                if start_match == -1:
                    start_match = i
                if index == len(sub):
                    return start_match, i + increment
            elif increment < 0 and start_match >= 0:
                index = 0
                i = start_match
                start_match = -1

            i += max(1, increment)

        return -1, -1

    def __is_last_result_ordered(self, results):
        """Indicates whether the last result in the results list is ordered. Unordered results are
        reference divs, footnotes, and any result that is preceded by a soft break that doesn't
        contain a new line."""

        result = results[-1]
        return not (result.type == "Div" and result.extras.get("identifier") == "refs" or
                    result.type == "Note" and result.extras.get("type") == "footnote" or
                    any(prev.type == "SoftBreak" and "\n" not in self.text[prev.start:prev.end]
                        for prev in results[:-1]))

    def __get_line(self, index):
        start_line = self.text.rfind("\n", 0, index) + 1
        end_line = self.text.find("\n", start_line)
        if end_line < 0:
            end_line = len(self.text)
        return start_line, end_line

    def __get_line_before(self, index):
        index = max(self.__get_line(index)[0] - 1, 0)
        return self.__get_line(index)

    def __get_line_after(self, index):
        index = min(self.__get_line(index)[1] + 1, len(self.text) - 1)
        return self.__get_line(index)
