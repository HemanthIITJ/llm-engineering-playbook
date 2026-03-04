import os

file_path = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\docs\llm_pretraining_playbook\01_strategic_justification.md"

with open(file_path, 'rb') as f:
    raw = f.read(20)
    print(f"First 20 bytes: {raw}")

# Let's also check the ones in llm_engineering_playbook to see if they're different
orig_file = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\llm_engineering_playbook\llm_pretraining_playbook\01_strategic_justification.md"
with open(orig_file, 'rb') as f:
    orig_raw = f.read(20)
    print(f"Original 20 bytes: {orig_raw}")
