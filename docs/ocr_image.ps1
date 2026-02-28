param([string]$ImagePath)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asbly = [System.Reflection.Assembly]::LoadWithPartialName("System.Runtime.WindowsRuntime")
$type = $asbly.GetType("System.WindowsRuntimeSystemExtensions")
$getAwaiter = $type.GetMethod("GetAwaiter")

Add-Type -AssemblyName Windows.Foundation
Add-Type -AssemblyName Windows.Graphics
Add-Type -AssemblyName Windows.Media

$localPath = Resolve-Path $ImagePath
$uri = New-Object System.Uri($localPath.Path)

$fileTask = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]::GetFileFromPathAsync($localPath.Path)
$awaitfile = $getAwaiter.MakeGenericMethod([Windows.Storage.StorageFile]).Invoke($null, @($fileTask))
while (-not $awaitfile.IsCompleted) { Start-Sleep -Milliseconds 10 }
$file = $awaitfile.GetResult()

$streamTask = $file.OpenAsync([Windows.Storage.FileAccessMode]::Read)
$awaitStream = $getAwaiter.MakeGenericMethod([Windows.Storage.Streams.IRandomAccessStream]).Invoke($null, @($streamTask))
while (-not $awaitStream.IsCompleted) { Start-Sleep -Milliseconds 10 }
$stream = $awaitStream.GetResult()

$decoderTask = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]::CreateAsync($stream)
$awaitDecoder = $getAwaiter.MakeGenericMethod([Windows.Graphics.Imaging.BitmapDecoder]).Invoke($null, @($decoderTask))
while (-not $awaitDecoder.IsCompleted) { Start-Sleep -Milliseconds 10 }
$decoder = $awaitDecoder.GetResult()

$bitmapTask = $decoder.GetSoftwareBitmapAsync()
$awaitBitmap = $getAwaiter.MakeGenericMethod([Windows.Graphics.Imaging.SoftwareBitmap]).Invoke($null, @($bitmapTask))
while (-not $awaitBitmap.IsCompleted) { Start-Sleep -Milliseconds 10 }
$bitmap = $awaitBitmap.GetResult()

# Ensure we have the right format for OCR
if ($bitmap.BitmapPixelFormat -ne [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8) {
    $bitmap = [Windows.Graphics.Imaging.SoftwareBitmap]::Convert($bitmap, [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8)
}

$engine = [Windows.Media.Ocr.OcrEngine, Windows.Media, ContentType = WindowsRuntime]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    Write-Output "OCR Engine creation failed."
    exit
}

$ocrTask = $engine.RecognizeAsync($bitmap)
$awaitOcr = $getAwaiter.MakeGenericMethod([Windows.Media.Ocr.OcrResult]).Invoke($null, @($ocrTask))
while (-not $awaitOcr.IsCompleted) { Start-Sleep -Milliseconds 10 }
$ocrResult = $awaitOcr.GetResult()

Write-Output $ocrResult.Text
