import os
import base64
import asyncio
import logging
import anthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from asyncio import sleep, Semaphore

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize the asynchronous client with the API key
client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=2)

async def read_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def read_prompt_from_file(prompt_filename):
    with open(prompt_filename, "r") as prompt_file:
        return prompt_file.read()

async def convert_image_to_markdown(semaphore: Semaphore, image_path, output_folder, model, prompt_filename, max_tokens):
    async with semaphore:
        image_basename = os.path.basename(image_path)
        markdown_filename = f"{os.path.splitext(image_basename)[0]}.md"
        output_path = os.path.join(output_folder, markdown_filename)
        
        if os.path.exists(output_path):
            logging.info(f"Markdown file {markdown_filename} already exists. Skipping.")
            return

        prompt = await read_prompt_from_file(prompt_filename)
        image_data = await read_image_as_base64(image_path)

        message_content = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[message_content,],
            )
        
            if hasattr(response, 'content') and len(response.content) > 0:
                for item in response.content:
                    if hasattr(item, 'type') and getattr(item, 'type') == 'text':  # Aangepast naar correcte benadering
                        markdown_text = getattr(item, 'text', '')
                        if markdown_text:
                            with open(output_path, "w") as markdown_file:
                                markdown_file.write(markdown_text)
                            logging.info(f"Markdown file saved at {output_path}")
                            return  # Stop na het verwerken van het eerste tekstblok
        
            else:
                logging.error("Response content is empty or not as expected.")
        
        except anthropic.RateLimitError as e:
            logging.info("Rate limit reached, waiting for 60 seconds.")
            await sleep(60)
            return await convert_image_to_markdown(semaphore, image_path, output_folder, model, prompt_filename, max_tokens)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            logging.error(f"API error: {e}")
            return

async def process_images_in_folder(input_folder, output_folder, model, prompt_filename, max_tokens):
    semaphore = Semaphore(6)  # Allow up to 6 simultaneous API calls
    images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    tasks = []
    for image in images:
        task = convert_image_to_markdown(semaphore, os.path.join(input_folder, image), output_folder, model, prompt_filename, max_tokens)
        tasks.append(task)

    await asyncio.gather(*tasks)

async def main():
    input_folder = "data/Owners_Manual_modelY_seperatepagejpgs"
    output_folder = "data/Owners_Manual_modelY_seperatepagemarkdowns"
    model = "claude-3-opus-20240229"
    prompt_filename = "Creating_a_vectorstore/01_DATAPREP/01a_DATAPREP_WITH_CLAUDEVISION/image_to_md_prompt.txt"
    max_tokens = 4096

    await process_images_in_folder(input_folder, output_folder, model, prompt_filename, max_tokens)

if __name__ == "__main__":
    asyncio.run(main())