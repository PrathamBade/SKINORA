import { useEffect, useRef, useState } from "react";

/**
 * CameraCapture component for Skinora
 * Handles camera stream acquisition, front/rear switching, visual face alignment guide,
 * and high-resolution snapshot capture. Ensures all media tracks are safely stopped on unmount.
 */
export default function CameraCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [facingMode, setFacingMode] = useState("user");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  // Check for multiple video input devices (front & back cameras)
  useEffect(() => {
    async function checkDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((d) => d.kind === "videoinput");
        setHasMultipleCameras(videoDevices.length > 1);
      } catch {
        // Non-fatal if enumeration is restricted
      }
    }
    checkDevices();
  }, []);

  // Start or restart camera stream
  useEffect(() => {
    let isCancelled = false;

    async function startCamera() {
      setCameraError("");
      setIsStreaming(false);

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError(
          "Camera access is not supported by your browser or connection. Please upload an image instead."
        );
        return;
      }

      // Stop previous tracks if switching facing mode
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      const constraints = {
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 960, max: 1080 },
        },
      };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (isCancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            if (!isCancelled) {
              videoRef.current.play().catch(() => {});
              setIsStreaming(true);
            }
          };
        }
      } catch (err) {
        if (isCancelled) return;
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          setCameraError(
            "Camera permission was denied. Please allow camera access in your browser settings or choose to upload an image."
          );
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          setCameraError(
            "No camera was detected on this device. Please choose to upload an image."
          );
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          setCameraError(
            "The camera is already in use by another application. Please close other camera apps and retry."
          );
        } else {
          setCameraError(
            "Could not access camera (" + (err.message || "Unknown error") + "). Please upload an image instead."
          );
        }
      }
    }

    startCamera();

    return () => {
      isCancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, [facingMode]);

  function switchCamera() {
    setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
  }

  function handleCapture() {
    const video = videoRef.current;
    if (!video || !isStreaming) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    // If using user/front-facing camera, mirror the capture to match what the user sees in the preview
    if (facingMode === "user") {
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, {
          type: "image/jpeg",
          lastModified: Date.now(),
        });
        // Stop stream before handing off
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
        }
        onCapture(file);
      },
      "image/jpeg",
      0.95
    );
  }

  return (
    <div className="camera-container" style={{ margin: "16px 0" }}>
      {cameraError ? (
        <div style={{ textAlign: "center", padding: "24px 16px" }}>
          <p style={{ color: "#ffb4a2", fontSize: "15px", marginBottom: "16px" }}>
            {cameraError}
          </p>
          <button className="btn btn-outline" onClick={onCancel}>
            Return to file upload
          </button>
        </div>
      ) : (
        <>
          <div className="camera-preview-wrapper">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-video"
              style={{
                transform: facingMode === "user" ? "scaleX(-1)" : "none",
              }}
            />
            {/* Visual Face Oval Alignment Guide */}
            <div className="camera-face-guide" />
            <div className="camera-guidance-pill">
              Position face within guide · Natural light · Steady
            </div>
          </div>

          <div className="camera-controls">
            <button
              type="button"
              className="btn btn-outline"
              onClick={onCancel}
              style={{ padding: "10px 16px", fontSize: "14px" }}
            >
              Cancel
            </button>

            <button
              type="button"
              className="btn-capture"
              onClick={handleCapture}
              disabled={!isStreaming}
              title="Capture Photo"
              aria-label="Capture Photo"
            >
              📷
            </button>

            {hasMultipleCameras && (
              <button
                type="button"
                className="btn btn-outline"
                onClick={switchCamera}
                title="Switch Camera (Front/Back)"
                style={{ padding: "10px 16px", fontSize: "14px" }}
              >
                🔄 Flip
              </button>
            )}
          </div>

          <p className="camera-notice">
            Stream is processed locally in your browser. No photo is stored or sent until confirmed.
          </p>
        </>
      )}
    </div>
  );
}

