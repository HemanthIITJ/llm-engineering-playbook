import shutil
import os

src = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\llm_engineering_playbook\llm_pretraining_playbook"
dst = r"c:\Users\heman\Desktop\code\ai_model_expermint\Blogs\llm-engineering-playbook\docs\llm_pretraining_playbook"

if os.path.exists(dst):
    shutil.rmtree(dst)
    
shutil.copytree(src, dst)
print("Successfully copied directory without corruption.")
