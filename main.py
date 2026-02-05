from fastapi import FastAPI, Header, HTTPException
import base64
import librosa
import numpy as np
import joblib

app = FastAPI()

API_KEY = "sk_test_123456789"

# Load trained model
model = joblib.load("voice_model.pkl")

@app.post("/api/voice-detection")
def detect_voice(payload: dict, x_api_key: str = Header(None)):
    import uuid
    request_id = str(uuid.uuid4())

    # 1. Required field validation
    required_fields = ["language", "audioFormat", "audioBase64"]

    for field in required_fields:
        if field not in payload:
            return {
                "status": "error",
                "message": f"Missing field: {field}"
            }
        
    # 2.Audio Format Validation
    if payload["audioFormat"].lower() != "mp3":
        return {
            "status": "error",
            "message": "Only mp3 format supported"
        }
    
    # 3. Language validation
    supported_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

    if payload.get("language") not in supported_languages:
        return {
            "status": "error",
            "message": "Unsupported language"
        }
    
    # 4. API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 5. Decode Base64 audio
    audio_b64 = payload["audioBase64"]
    import binascii

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except binascii.Error:
        return {
            "status": "error",
            "message": "Invalid Base64 audio"
        }
    
    # 6. File Size Validation (max 5MB)
    if len(audio_bytes) > 5 * 1024 * 1024:
        return {
            "status": "error",
            "message": "Audio file too large"
        }

    # 7. Save temp file
    with open("test.mp3", "wb") as f:
        f.write(audio_bytes)

    # 8. Audio duration check 
    y, sr = librosa.load("test.mp3", sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < 2:
        return {
            "status": "error",
            "message": "Audio too short for reliable detection"
        }

    # 9. Load audio
    y, sr = librosa.load("test.mp3", sr=16000)

    # 10. Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # 11. Predict using trained model
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0].max()

    if pred == 1:
        classification = "AI_GENERATED"
        explanation = f"Model detected synthetic acoustic patterns with confidence {round(prob,2)}"
    else:
        classification = "HUMAN"
        explanation = f"Model detected natural human speech variations with confidence {round(prob,2)}"

    # 12. Logging
    with open("logs.txt", "a") as f:
        f.write(f"{payload['language']} -> {classification} ({round(prob,2)})\n")
    
    # 13. Return response
    return {
        "status": "success",
        "requestId": request_id,
        "language": payload["language"],
        "classification": classification,
        "confidenceScore": round(prob, 2),
        "confidenceLabel": "HIGH" if prob > 0.80 else "MEDIUM" if prob > 0.50 else "LOW",
        "audioDuration": round(duration, 2),
        "mfccMean": float(np.mean(features)),
        "explanation": explanation,

    }
