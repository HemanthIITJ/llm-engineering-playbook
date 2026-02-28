import asyncio
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder, BitmapPixelFormat
from winrt.windows.storage import StorageFile

async def ocr(path):
    try:
        file = await StorageFile.get_file_from_path_async(path)
        stream = await file.open_async(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        if bitmap.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
            import winrt.windows.graphics.imaging as imaging
            bitmap = imaging.SoftwareBitmap.convert(bitmap, BitmapPixelFormat.BGRA8)
        engine = OcrEngine.try_create_from_user_profile_languages()
        result = await engine.recognize_async(bitmap)
        print("Success:")
        print(result.text)
    except Exception as e:
        print("Error:", e)

asyncio.run(ocr(r"C:\Users\heman\Desktop\code\ai_model_expermint\Blogs\Cheat_book\assets\LLM_Evolution_and_Scaling_page_1.png"))
