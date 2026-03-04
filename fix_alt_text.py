import os
import re

directory = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\llm_engineering_playbook"

count = 0
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".md"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            def replace_alt(match):
                alt_text = match.group(1).strip()
                image_path = match.group(2)
                # If alt text is empty or too short, replace it with filename logic
                if len(alt_text) < 3:
                    filename = os.path.basename(image_path)
                    name_without_ext = os.path.splitext(filename)[0]
                    # Remove percentage encoding and convert to spaces
                    import urllib.parse
                    decoded = urllib.parse.unquote(name_without_ext)
                    new_alt = decoded.replace("_", " ").replace("-", " ").strip()
                    if len(new_alt) < 3:
                        new_alt = "illustration"
                    return f"![{new_alt}]({image_path})"
                return match.group(0)
            
            new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_alt, content)
            
            # Optionally check for raw HTML <img> tags with missing alt text
            def replace_html_img(match):
                img_tag = match.group(0)
                if 'alt="' not in img_tag and "alt='" not in img_tag:
                    src_match = re.search(r'src=["\']([^"\']+)["\']', img_tag)
                    if src_match:
                        filename = os.path.basename(src_match.group(1))
                        name_without_ext = os.path.splitext(filename)[0]
                        new_alt = urllib.parse.unquote(name_without_ext).replace("_", " ").replace("-", " ").strip()
                        return img_tag.replace('<img ', f'<img alt="{new_alt}" ')
                return img_tag

            new_content = re.sub(r'<img\b[^>]*>', replace_html_img, new_content)

            if new_content != content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated {filepath}")
                count += 1

print(f"Total files updated: {count}")
