import re

def strip_language_details(content: str, target_language: str) -> str:
    """
    Strip out <details> tags for non-target language sections and remove the tags
    for the target language while preserving its content.
    
    Args:
        content: The README content
        target_language: Either 'Python' or 'TypeScript'
    """
    # Define the opposite language to remove
    opposite_language = "TypeScript" if target_language == "Python" else "Python"
    
    # First remove the opposite language blocks completely
    pattern = rf'<details[^>]*>\s*<summary>{opposite_language}</summary>.*?</details>'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Then remove just the detail/summary tags for target language, keeping content
    pattern = rf'<details[^>]*>\s*<summary>{target_language}</summary>(.*?)</details>'
    
    def replace_match(match):
        return match.group(1).strip()
    
    content = re.sub(pattern, replace_match, content, flags=re.DOTALL)
    
    # Clean up any double newlines created during the process
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content

def main():
    # Read the README
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Generate Python version
    python_content = strip_language_details(content, "Python")
    with open('./python/README.md', 'w') as f:
        f.write(python_content)
    
    # Generate TypeScript version
    ts_content = strip_language_details(content, "TypeScript")
    with open('./js/README.md', 'w') as f:
        f.write(ts_content)

if __name__ == "__main__":
    main()
