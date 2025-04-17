import re
import emoji

def prepare_tts_input_with_context(text: str) -> str:
    """
    Prepares text for a TTS API by cleaning Markdown and adding minimal contextual hints
    for certain Markdown elements like headers. Preserves paragraph separation.

    Args:
        text (str): The raw text containing Markdown or other formatting.

    Returns:
        str: Cleaned text with contextual hints suitable for TTS input.
    """

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Add context for headers
    def header_replacer(match):
        level = len(match.group(1))  # Number of '#' symbols
        header_text = match.group(2).strip()
        if level == 1:
            return f"Title — {header_text}\n"
        elif level == 2:
            return f"Section — {header_text}\n"
        else:
            return f"Subsection — {header_text}\n"

    text = re.sub(r"^(#{1,6})\s+(.*)", header_replacer, text, flags=re.MULTILINE)

    # Announce links (currently commented out for potential future use)
    # text = re.sub(r"\[([^\]]+)\]\((https?:\/\/[^\)]+)\)", r"\1 (link: \2)", text)

    # Remove links while keeping the link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Describe inline code
    text = re.sub(r"`([^`]+)`", r"code snippet: \1", text)

    # Remove bold/italic symbols but keep the content
    text = re.sub(r"(\*\*|__|\*|_)", '', text)

    # Remove code blocks (multi-line) with a description
    text = re.sub(r"```([\s\S]+?)```", r"(code block omitted)", text)

    # Remove image syntax but add alt text if available
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"Image: \1", text)

    # Remove HTML tags
    text = re.sub(r"</?[^>]+(>|$)", '', text)

    # Normalize line breaks
    text = re.sub(r"\n{2,}", '\n\n', text)  # Ensure consistent paragraph separation

    # Replace multiple spaces within lines
    text = re.sub(r" {2,}", ' ', text)

    # Trim leading and trailing whitespace from the whole text
    text = text.strip()

    return text
