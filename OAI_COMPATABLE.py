import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatterbox.models.s3gen import S3GEN_SR
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

tts_model: Optional[ChatterboxTTS] = None
vc_model: Optional[ChatterboxVC] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

_audio_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model loading and cleanup."""
    global tts_model, vc_model
    
    logger.info(f"Loading models on device: {device}")
    
    try:
        logger.info("Loading ChatterboxTTS model...")
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info("ChatterboxTTS model loaded successfully")
        
        logger.info("Loading ChatterboxVC model...")
        vc_model = ChatterboxVC.from_pretrained(device=device)
        logger.info("ChatterboxVC model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

    
    yield
    
    logger.info("Cleaning up models and releasing resources")
    if tts_model is not None:
        if device == "cuda":
            torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    
    class Config:
        validate_assignment = True
        
    def __init__(self, **data):
        super().__init__(**data)
        if len(self.text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        if len(self.text) > 1000:
            raise ValueError("Text too long (max 1000 characters)")
        
        if not 0.1 <= self.exaggeration <= 3.0:
            raise ValueError("Exaggeration must be between 0.1 and 3.0")
        if not 0.1 <= self.cfg_weight <= 1.0:
            raise ValueError("CFG weight must be between 0.1 and 1.0")
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.1 and 2.0")


async def process_audio_upload(
    audio_file: UploadFile, target_sr: int, description: str
) -> tuple[str, float]:
    """
    Process uploaded audio file and return temporary file path and duration.
    
    Args:
        audio_file: Uploaded audio file
        target_sr: Target sample rate for resampling
        description: Description for logging
        
    Returns:
        Tuple of (temp_file_path, audio_duration)
    """
    try:
        if audio_file.size and audio_file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Audio file too large (max 50MB)")
        
        if audio_file.content_type and not audio_file.content_type.startswith('audio/'):
            logger.warning(f"Unexpected content type: {audio_file.content_type}")
        
        content = await audio_file.read()
        
        try:
            y, sr = librosa.load(io.BytesIO(content), sr=None, duration=30.0)  # Limit to 30 seconds
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode audio file: {e}")
        
        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty or corrupted")
        
        duration = len(y) / sr
        if duration < 0.5:
            raise HTTPException(status_code=400, detail="Audio too short (minimum 0.5 seconds)")
        if duration > 30.0:
            logger.warning(f"Audio duration {duration:.2f}s exceeds recommended 30s")
        
        if sr != target_sr:
            logger.info(f"Resampling {description} from {sr}Hz to {target_sr}Hz")
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        if y.max() > 0:
            y = y / max(abs(y.max()), abs(y.min()))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, y, target_sr)
            temp_path = tmp_file.name
        
        logger.info(f"Processed {description} (WAV, {target_sr}Hz, {duration:.2f}s): {temp_path}")
        
        return temp_path, duration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {description}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")


def create_audio_response(wav_tensor: torch.Tensor, sample_rate: int) -> StreamingResponse:
    """Create audio response from tensor with memory optimization."""
    try:
        wav_numpy = wav_tensor.squeeze().detach().cpu().numpy()
        
        if device == "cuda":
            del wav_tensor
            torch.cuda.empty_cache()
        
        buffer = io.BytesIO()
        sf.write(buffer, wav_numpy, samplerate=sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_audio.wav",
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logger.error(f"Error creating audio response: {e}")
        raise HTTPException(status_code=500, detail="Failed to create audio response")


def log_performance_metrics(start_time: float, wav_tensor: torch.Tensor, sample_rate: int, task: str):
    """Log performance metrics for generation tasks."""
    try:
        processing_time = time.time() - start_time
        wav_numpy = wav_tensor.squeeze().detach().cpu().numpy()
        audio_duration = len(wav_numpy) / sample_rate
        
        if audio_duration > 0:
            rtf = processing_time / audio_duration
            logger.info(f"{task} - Processing: {processing_time:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.2f}")
        else:
            logger.info(f"{task} - Processing: {processing_time:.2f}s, Audio: 0s (Cannot calculate RTF)")
            
        if device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            logger.info(f"{task} - GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
    except Exception as e:
        logger.warning(f"Error logging performance metrics: {e}")

@app.post("/v1/audio/speech")
async def text_to_speech(
    text_payload_str: str = Form(..., alias="text_payload"),
    audio_prompt: UploadFile = File(None),
):
    """Generate speech from text with optional audio prompt for voice cloning."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is not loaded")

    audio_prompt_path_temp = None
    request_start_time = time.time()
    
    try:
        try:
            if text_payload_str.startswith('{'):
                import json
                payload_dict = json.loads(text_payload_str)
                text_payload = TTSRequest(**payload_dict)
            else:
                text_payload = TTSRequest.model_validate_json(text_payload_str)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid text_payload format: {e}")

        logger.info(f"TTS request - Text length: {len(text_payload.text)} chars, Exaggeration: {text_payload.exaggeration}")

        if audio_prompt and audio_prompt.filename:
            logger.info(f"Processing audio prompt: {audio_prompt.filename}")
            audio_prompt_path_temp, prompt_duration = await process_audio_upload(
                audio_prompt, S3_SR, "audio prompt"
            )
            
            cond_start = time.time()
            tts_model.prepare_conditionals(
                audio_prompt_path_temp,
                exaggeration=text_payload.exaggeration
            )
            logger.info(f"Conditionals prepared in {time.time() - cond_start:.2f}s")
            
        elif tts_model.conds is None:
            raise HTTPException(
                status_code=400,
                detail="No audio prompt provided and no default voice loaded. "
                       "Please provide an audio_prompt or ensure a default voice is available."
            )

        logger.info(f"Generating speech for text: '{text_payload.text[:50]}{'...' if len(text_payload.text) > 50 else ''}'")
        generation_start = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            wav_tensor = tts_model.generate(
                text=text_payload.text,
                exaggeration=text_payload.exaggeration,
                cfg_weight=text_payload.cfg_weight,
                temperature=text_payload.temperature,
            )
        
        log_performance_metrics(generation_start, wav_tensor, tts_model.sr, "TTS")
        
        total_time = time.time() - request_start_time
        logger.info(f"Total TTS request time: {total_time:.2f}s")
        
        return create_audio_response(wav_tensor, tts_model.sr)

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during TTS generation")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="Insufficient GPU memory for generation")
    except Exception as e:
        logger.error(f"Error during TTS generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_prompt_path_temp and os.path.exists(audio_prompt_path_temp):
            try:
                os.remove(audio_prompt_path_temp)
                logger.debug(f"Removed temporary audio prompt: {audio_prompt_path_temp}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {audio_prompt_path_temp}: {e}")


@app.post("/v1/audio/conversions")
async def voice_conversion(
    source_audio: UploadFile = File(...),
    target_voice: UploadFile = File(...),
):
    """Convert source audio to target voice using voice conversion."""
    if vc_model is None:
        raise HTTPException(status_code=503, detail="VC model is not loaded")

    source_audio_path_temp = None
    target_voice_path_temp = None
    request_start_time = time.time()
    
    try:
        if not source_audio.filename:
            raise HTTPException(status_code=400, detail="Source audio file is required")
        if not target_voice.filename:
            raise HTTPException(status_code=400, detail="Target voice file is required")

        logger.info(f"VC request - Source: {source_audio.filename}, Target: {target_voice.filename}")

        source_audio_path_temp, source_duration = await process_audio_upload(
            source_audio, S3_SR, "source audio"
        )

        target_voice_path_temp, target_duration = await process_audio_upload(
            target_voice, S3GEN_SR, "target voice"
        )
        
        target_setup_start = time.time()
        vc_model.set_target_voice(target_voice_path_temp)
        logger.info(f"Target voice setup completed in {time.time() - target_setup_start:.2f}s")

        logger.info(f"Performing voice conversion (source: {source_duration:.2f}s, target: {target_duration:.2f}s)")
        generation_start = time.time()
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            wav_tensor = vc_model.generate(audio=source_audio_path_temp)
        
        log_performance_metrics(generation_start, wav_tensor, vc_model.sr, "VC")
        
        total_time = time.time() - request_start_time
        logger.info(f"Total VC request time: {total_time:.2f}s")
        
        return create_audio_response(wav_tensor, vc_model.sr)

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during voice conversion")
        if device == "cuda":
            torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="Insufficient GPU memory for conversion")
    except Exception as e:
        logger.error(f"Error during voice conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_path, description in [
            (source_audio_path_temp, "source audio"),
            (target_voice_path_temp, "target voice")
        ]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Removed temporary {description}: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary {description} file {temp_path}: {e}")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": device,
        "models_loaded": {
            "tts": tts_model is not None,
            "vc": vc_model is not None
        }
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information."""
    gpu_info = {}
    if device == "cuda" and torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": device,
        "models_loaded": {
            "tts": tts_model is not None,
            "vc": vc_model is not None
        },
        "gpu_info": gpu_info,
        "sample_rates": {
            "s3_sr": S3_SR,
            "s3gen_sr": S3GEN_SR
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8989"))
    log_level = os.getenv("LOG_LEVEL", "info")
    workers = int(os.getenv("WORKERS", "1"))
    logger.info(f"Starting Chatterbox API server on {host}:{port}")
    logger.info(f"Device: {device}")
    logger.info(f"Log level: {log_level}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
        workers=workers,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )