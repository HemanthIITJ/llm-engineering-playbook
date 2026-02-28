import os
import asyncio
import re
import fitz  # PyMuPDF

import winrt.windows.graphics.imaging as imaging
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage import StorageFile

async def get_text_from_image(path):
    try:
        file = await StorageFile.get_file_from_path_async(path)
        stream = await file.open_async(0)
        decoder = await imaging.BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        if bitmap.bitmap_pixel_format != imaging.BitmapPixelFormat.BGRA8:
            bitmap = imaging.SoftwareBitmap.convert(bitmap, imaging.BitmapPixelFormat.BGRA8)
        engine = OcrEngine.try_create_from_user_profile_languages()
        result = await engine.recognize_async(bitmap)
        return result.text
    except Exception as e:
        print(f"OCR Error on {path}: {e}")
        return ""

def get_words(text):
    return set(re.findall(r'\b\w{4,}\b', text.lower()))

async def process_pdfs_async():
    base_dir = r"C:\Users\heman\Desktop\code\ai_model_expermint\Blogs\Cheat_book"
    notes_dir = os.path.join(base_dir, "notes")
    assets_dir = os.path.join(base_dir, "assets")
    
    mapping = {
        "Augmented_LLM_Evolution": "ALMs.md",
        "Efficient_LLM_Scaling": "PEFT.md",
        "LLM_Reasoning_Topologies": "Prompting.md",
        "Universal_LLM_Frontiers": "Multimodal.md",
        "System_2_LLM_Engineering": "Recent.md"
    }
    
    for pdf_basename, md_name in mapping.items():
        pdf_path = os.path.join(notes_dir, f"{pdf_basename}.pdf")
        md_path = os.path.join(base_dir, md_name)
        if not os.path.exists(pdf_path) or not os.path.exists(md_path):
            print(f"Skipping {pdf_basename} because PDF or MD is missing.")
            continue
            
        print(f"\nProcessing matching for {pdf_basename} -> {md_name}")
        
        # 1. Convert PDF to PNGs first
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        for page_num in range(num_pages):
            png_name = f"{pdf_basename}_page_{page_num + 1}.png"
            png_path = os.path.join(assets_dir, png_name)
            if not os.path.exists(png_path):
                print(f"  Generating PNG for page {page_num + 1}...")
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) # High resolution
                pix.save(png_path)
        doc.close()
        
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Clean up old bulk inserts
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if f"![](assets/{pdf_basename}" in line:
                i += 1
                continue
            new_lines.append(line)
            i += 1
            
        lines = new_lines
        
        # Build chunks of text (e.g. paragraphs or 10 lines at a time) to match against
        chunks = [] # (start_idx, set_of_words, text)
        chunk_lines = []
        chunk_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == "": # paragraph break
                if chunk_lines:
                    text_block = " ".join(chunk_lines)
                    words = get_words(text_block)
                    if words:
                        chunks.append((chunk_start, words, text_block))
                chunk_lines = []
                chunk_start = i + 1
            else:
                if not chunk_lines:
                    chunk_start = i
                chunk_lines.append(line)
                
        if chunk_lines:
            text_block = " ".join(chunk_lines)
            chunks.append((chunk_start, get_words(text_block), text_block))
            
        insertions = []
        
        last_inserted_idx = 0
        for page_num in range(1, num_pages + 1):
            png_name = f"{pdf_basename}_page_{page_num}.png"
            png_path = os.path.join(assets_dir, png_name)
            
            # Extract text via OCR
            text = await get_text_from_image(png_path)
            ocr_words = get_words(text)
            
            img_syntax = f"![](assets/{png_name})\n"
            
            best_match_idx = -1
            best_score = 0.0
            
            if ocr_words:
                for start_idx, chunk_words, chunk_text in chunks:
                    # Enforce sequential logic slightly: punish going backwards too much
                    if start_idx < last_inserted_idx - 50:
                        continue
                        
                    intersection = ocr_words.intersection(chunk_words)
                    score = len(intersection) / (len(ocr_words) + 1)
                    
                    if score > best_score:
                        best_score = score
                        best_match_idx = start_idx
                        
            if best_score > 0.1 and best_match_idx != -1:
                print(f"  Page {page_num} -> Matched chunk at line {best_match_idx} (Score: {best_score:.2f})")
                insertions.append((best_match_idx, img_syntax))
                last_inserted_idx = best_match_idx
            else:
                # If no chunk matches, just put it sequentially after the last one
                fallback_idx = min(last_inserted_idx + 10, len(lines))
                print(f"  Page {page_num} -> No strong match. Fallback to line {fallback_idx}")
                insertions.append((fallback_idx, img_syntax))
                last_inserted_idx = fallback_idx

        # Apply insertions in reverse order to avoid messing up indices
        insertions.sort(key=lambda x: x[0], reverse=True)
        for idx, insert_text in insertions:
            lines.insert(idx, "\n" + insert_text + "\n")
            
        with open(md_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
if __name__ == "__main__":
    asyncio.run(process_pdfs_async())
