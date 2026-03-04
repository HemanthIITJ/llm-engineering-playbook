import os

def convert_to_utf8(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                
                # First try to read as utf-16 (common culprit for these characters)
                try:
                    with open(filepath, 'r', encoding='utf-16') as f:
                        content = f.read()
                    # If successful, it was utf-16. Write back as utf-8
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Converted {file} from UTF-16 to UTF-8")
                    count += 1
                    continue
                except UnicodeError:
                    pass
                
                # If not utf-16, try Windows-1252 or utf-8-sig (BOM)
                try:
                    with open(filepath, 'rb') as f:
                        raw = f.read()
                    if raw.startswith(b'\xef\xbb\xbf'):
                        content = raw.decode('utf-8-sig')
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Removed BOM from {file}")
                        count += 1
                except Exception as e:
                    print(f"Error checking {file}: {e}")

    print(f"Total files converted: {count}")

docs_dir = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\docs\llm_pretraining_playbook"
convert_to_utf8(docs_dir)
