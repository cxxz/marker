import re
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from marker.settings import settings
from anthropic import AnthropicBedrock

load_dotenv()

client = AnthropicBedrock(
    aws_region="us-west-2",
)

def sort_table_blocks(blocks, tolerance=5):
    vertical_groups = {}
    for block in blocks:
        if hasattr(block, "bbox"):
            bbox = block.bbox
        else:
            bbox = block["bbox"]
        group_key = round(bbox[1] / tolerance)
        if group_key not in vertical_groups:
            vertical_groups[group_key] = []
        vertical_groups[group_key].append(block)

    # Sort each group horizontally and flatten the groups into a single list
    sorted_blocks = []
    for _, group in sorted(vertical_groups.items()):
        sorted_group = sorted(group, key=lambda x: x.bbox[0] if hasattr(x, "bbox") else x["bbox"][0])
        sorted_blocks.extend(sorted_group)

    return sorted_blocks


def replace_dots(text):
    dot_pattern = re.compile(r'(\s*\.\s*){4,}')
    dot_multiline_pattern = re.compile(r'.*(\s*\.\s*){4,}.*', re.DOTALL)

    if dot_multiline_pattern.match(text):
        text = dot_pattern.sub(' ', text)
    return text


def replace_newlines(text):
    # Replace all newlines
    newline_pattern = re.compile(r'[\r\n]+')
    return newline_pattern.sub(' ', text.strip())


def pil_image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64 string.

    :param pil_image: PIL Image object to be converted.
    :return: Base64 string of the image.
    """
    # Create a BytesIO buffer
    buffered = BytesIO()

    # Save the image to the buffer in PNG format
    pil_image.save(buffered, format="PNG")

    # Get the byte data from the buffer
    image_data = buffered.getvalue()

    # Encode the byte data to base64
    return base64.b64encode(image_data).decode('utf-8')


def markdown_table_image(table_image, model = settings.LVM_MODEL):
    image_data = pil_image_to_base64(table_image)
    print(f"CONG TEST sending table image to LLM")
    llm_response = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Convert the table in the image to Markdown format.",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                ],
            },
        ],
        model=model,
        temperature=0.1,
    )
    return llm_response.content[0].text
