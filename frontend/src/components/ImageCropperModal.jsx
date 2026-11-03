import { useEffect, useRef, useState } from "react";

/**
 * ImageCropperModal component for Skinora
 * Provides interactive client-side face cropping with pan, zoom, 1:1 aspect ratio locking,
 * optional browser FaceDetector suggestion, and high-resolution JPEG export.
 */
export default function ImageCropperModal({ file, onConfirmCrop, onCancel }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  
  // Crop box state in normalized coordinates [0, 1] relative to the original image
  const [cropBox, setCropBox] = useState({ x: 0.1, y: 0.1, size: 0.8 });
  const [zoom, setZoom] = useState(1.0);
  const [lockSquare, setLockSquare] = useState(true);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Dragging / Resizing interaction state
  const dragRef = useRef({
    isDragging: false,
    isResizing: false,
    startX: 0,
    startY: 0,
    startBox: null,
    resizeCorner: null,
  });

  // Load image object from input File
  useEffect(() => {
    if (!file) return;
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.src = objectUrl;

    img.onload = async () => {
      imageRef.current = img;
      setImageLoaded(true);

      // Attempt optional lightweight client-side face detection suggestion
      let detectedBox = null;
      if ("FaceDetector" in window) {
        try {
          const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
          const faces = await detector.detect(img);
          if (faces && faces.length > 0) {
            const f = faces[0].boundingBox;
            // Add comfortable margin around detected face (35% padding)
            const margin = Math.max(f.width, f.height) * 0.35;
            const cx = f.x + f.width / 2;
            const cy = f.y + f.height / 2;
            const sizePx = Math.max(f.width, f.height) + margin * 2;
            
            // Normalize to [0, 1]
            const minDim = Math.min(img.width, img.height);
            const clampedSize = Math.min(sizePx, minDim);
            const normSize = clampedSize / minDim;
            const normX = Math.max(0, Math.min(1 - normSize, (cx - clampedSize / 2) / img.width));
            const normY = Math.max(0, Math.min(1 - normSize, (cy - clampedSize / 2) / img.height));

            detectedBox = { x: normX, y: normY, size: normSize };
            setIsFaceDetected(true);
          }
        } catch {
          // Fall back to centered crop
        }
      }

      if (!detectedBox) {
        // Default: Centered square crop covering 80% of smallest dimension
        const aspect = img.width / img.height;
        if (aspect >= 1) {
          // Wider than tall
          const sizeNorm = 0.85;
          const xNorm = (1 - sizeNorm * (img.height / img.width)) / 2;
          setCropBox({ x: Math.max(0, xNorm), y: 0.075, size: sizeNorm });
        } else {
          // Taller than wide
          const sizeNorm = 0.85;
          const yNorm = (1 - sizeNorm * (img.width / img.height)) / 2;
          setCropBox({ x: 0.075, y: Math.max(0, yNorm), size: sizeNorm });
        }
      } else {
        setCropBox(detectedBox);
      }
    };

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  // Redraw canvas with image and interactive crop overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !imageLoaded) return;

    const ctx = canvas.getContext("2d");
    const containerWidth = Math.min(540, window.innerWidth - 60);
    const containerHeight = 360;

    // Compute display dimensions preserving aspect ratio
    const imgAspect = img.width / img.height;
    let dispW = containerWidth;
    let dispH = containerWidth / imgAspect;
    if (dispH > containerHeight) {
      dispH = containerHeight;
      dispW = containerHeight * imgAspect;
    }

    canvas.width = dispW;
    canvas.height = dispH;

    // Clear
    ctx.clearRect(0, 0, dispW, dispH);

    // Draw base image
    ctx.drawImage(img, 0, 0, dispW, dispH);

    // Dark semi-transparent mask over non-cropped areas
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fillRect(0, 0, dispW, dispH);

    // Calculate crop rectangle in display pixels
    const cropX = cropBox.x * dispW;
    const cropY = cropBox.y * dispH;
    const cropSize = cropBox.size * Math.min(dispW, dispH) * zoom;
    const cropW = cropSize;
    const cropH = cropSize;

    // Reveal cropped region
    ctx.save();
    ctx.beginPath();
    ctx.rect(cropX, cropY, cropW, cropH);
    ctx.clip();
    ctx.drawImage(img, 0, 0, dispW, dispH);
    ctx.restore();

    // Draw crop border
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(cropX, cropY, cropW, cropH);

    // Draw rule-of-thirds grid inside crop
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    // vertical lines
    ctx.moveTo(cropX + cropW / 3, cropY);
    ctx.lineTo(cropX + cropW / 3, cropY + cropH);
    ctx.moveTo(cropX + (2 * cropW) / 3, cropY);
    ctx.lineTo(cropX + (2 * cropW) / 3, cropY + cropH);
    // horizontal lines
    ctx.moveTo(cropX, cropY + cropH / 3);
    ctx.lineTo(cropX + cropW, cropY + cropH / 3);
    ctx.moveTo(cropX, cropY + (2 * cropH) / 3);
    ctx.lineTo(cropX + cropW, cropY + (2 * cropH) / 3);
    ctx.stroke();

    // Draw corner handles
    const handleSize = 10;
    ctx.fillStyle = "#edd3ba";
    ctx.strokeStyle = "#27382c";
    ctx.lineWidth = 1.5;

    const corners = [
      [cropX, cropY],
      [cropX + cropW, cropY],
      [cropX, cropY + cropH],
      [cropX + cropW, cropY + cropH],
    ];

    corners.forEach(([cx, cy]) => {
      ctx.fillRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
      ctx.strokeRect(cx - handleSize / 2, cy - handleSize / 2, handleSize, handleSize);
    });
  }, [cropBox, zoom, imageLoaded]);

  // Pointer / Touch interaction handlers
  function getCanvasCoords(e) {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  function handlePointerDown(e) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { x, y } = getCanvasCoords(e);

    const dispW = canvas.width;
    const dispH = canvas.height;
    const cropX = cropBox.x * dispW;
    const cropY = cropBox.y * dispH;
    const cropSize = cropBox.size * Math.min(dispW, dispH) * zoom;

    // Check if clicked near bottom-right corner for resizing
    const hitRadius = 24;
    const brX = cropX + cropSize;
    const brY = cropY + cropSize;

    if (Math.abs(x - brX) < hitRadius && Math.abs(y - brY) < hitRadius) {
      dragRef.current = {
        isDragging: false,
        isResizing: true,
        startX: x,
        startY: y,
        startBox: { ...cropBox },
      };
      return;
    }

    // Check if clicked inside crop box for dragging
    if (x >= cropX && x <= cropX + cropSize && y >= cropY && y <= cropY + cropSize) {
      dragRef.current = {
        isDragging: true,
        isResizing: false,
        startX: x,
        startY: y,
        startBox: { ...cropBox },
      };
    }
  }

  function handlePointerMove(e) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { isDragging, isResizing, startX, startY, startBox } = dragRef.current;
    if (!isDragging && !isResizing) return;

    const { x, y } = getCanvasCoords(e);
    const dispW = canvas.width;
    const dispH = canvas.height;
    const minDim = Math.min(dispW, dispH);

    if (isDragging) {
      const dx = (x - startX) / dispW;
      const dy = (y - startY) / dispH;
      const effectiveSize = startBox.size * (minDim / dispW) * zoom;
      const effectiveSizeY = startBox.size * (minDim / dispH) * zoom;

      const newX = Math.max(0, Math.min(1 - effectiveSize, startBox.x + dx));
      const newY = Math.max(0, Math.min(1 - effectiveSizeY, startBox.y + dy));

      setCropBox((prev) => ({ ...prev, x: newX, y: newY }));
    } else if (isResizing) {
      const delta = (x - startX) / minDim;
      const newSize = Math.max(0.25, Math.min(1.0, startBox.size + delta));
      setCropBox((prev) => ({ ...prev, size: newSize }));
    }
  }

  function handlePointerUp() {
    dragRef.current.isDragging = false;
    dragRef.current.isResizing = false;
  }

  function resetCrop() {
    setZoom(1.0);
    setCropBox({ x: 0.1, y: 0.1, size: 0.8 });
  }

  function confirmCrop() {
    const img = imageRef.current;
    if (!img) return;

    // Map normalized cropBox coordinates to original image pixel dimensions
    const minImgDim = Math.min(img.width, img.height);
    const cropSizePx = Math.round(cropBox.size * minImgDim * zoom);
    const cropXPx = Math.round(cropBox.x * img.width);
    const cropYPx = Math.round(cropBox.y * img.height);

    // Safeguard bounds
    const safeW = Math.min(cropSizePx, img.width - cropXPx);
    const safeH = Math.min(cropSizePx, img.height - cropYPx);
    const finalDim = Math.max(224, Math.min(safeW, safeH));

    // Render cropped image to off-screen export canvas
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = finalDim;
    exportCanvas.height = finalDim;
    const ctx = exportCanvas.getContext("2d");

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, cropXPx, cropYPx, safeW, safeH, 0, 0, finalDim, finalDim);

    exportCanvas.toBlob(
      (blob) => {
        if (!blob) return;
        const croppedFile = new File([blob], `cropped_${file.name.replace(/\.[^/.]+$/, "")}.jpg`, {
          type: "image/jpeg",
          lastModified: Date.now(),
        });
        onConfirmCrop(croppedFile);
      },
      "image/jpeg",
      0.95
    );
  }

  return (
    <div className="cropper-modal-overlay">
      <div className="cropper-modal">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h2>Adjust & Crop Facial Image</h2>
            <p>
              Focus on the central facial region for optimal AI acne assessment accuracy.
            </p>
          </div>
          {isFaceDetected && (
            <span
              style={{
                background: "#e7f4e9",
                color: "#216128",
                padding: "4px 10px",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: 600,
              }}
            >
              ✓ Face Centered
            </span>
          )}
        </div>

        <div className="cropper-canvas-wrapper">
          <canvas
            ref={canvasRef}
            className="cropper-canvas"
            onMouseDown={handlePointerDown}
            onMouseMove={handlePointerMove}
            onMouseUp={handlePointerUp}
            onMouseLeave={handlePointerUp}
            onTouchStart={handlePointerDown}
            onTouchMove={handlePointerMove}
            onTouchEnd={handlePointerUp}
            style={{ cursor: "grab" }}
          />
        </div>

        {/* Zoom Slider Control */}
        <div style={{ margin: "16px 0 8px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "14px", color: "var(--text-muted)" }}>
            <span>Zoom Crop Region</span>
            <span>{Math.round(zoom * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.7"
            max="1.5"
            step="0.05"
            value={zoom}
            onChange={(e) => setZoom(parseFloat(e.target.value))}
            style={{ width: "100%", accentColor: "var(--bg-accent-sage)", margin: "6px 0" }}
          />
        </div>

        <div style={{ fontSize: "13px", color: "var(--text-subtle)", marginBottom: "16px" }}>
          💡 Drag inside the box to reposition. Drag the bottom-right corner to resize.
        </div>

        <div className="cropper-actions">
          <button type="button" className="btn btn-outline" onClick={resetCrop}>
            Reset Box
          </button>

          <div className="cropper-btn-group">
            <button type="button" className="btn btn-outline" onClick={onCancel}>
              Use Original
            </button>
            <button type="button" className="btn btn-primary" onClick={confirmCrop}>
              ✓ Confirm Crop
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

