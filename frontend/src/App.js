import { useState, useEffect, useRef, useCallback } from "react";
import "@/App.css";
import axios from "axios";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Spirit Box word banks for random "transmissions"
const SPIRIT_WORDS = [
  "hello", "help", "here", "yes", "no", "leave", "stay", "danger",
  "follow", "behind", "cold", "dark", "light", "death", "life",
  "lost", "find", "home", "gone", "watch", "listen", "quiet",
  "run", "stop", "wait", "soon", "never", "always", "fear"
];

function App() {
  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const lastAlertTime = useRef(0);

  // State
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [detectionCount, setDetectionCount] = useState(0);
  const [emfLevel, setEmfLevel] = useState(0);
  const [spiritBoxFreq, setSpiritBoxFreq] = useState(88.1);
  const [spiritBoxActive, setSpiritBoxActive] = useState(false);
  const [spiritWord, setSpiritWord] = useState("");
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [alertMessage, setAlertMessage] = useState("");
  const [currentPoses, setCurrentPoses] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  // Initialize TensorFlow and Pose Detection
  const initializeDetector = useCallback(async () => {
    setIsLoading(true);
    try {
      await tf.ready();
      await tf.setBackend("webgl");
      
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        }
      );
      detectorRef.current = detector;
      console.log("Pose detector initialized");
    } catch (error) {
      console.error("Error initializing detector:", error);
    }
    setIsLoading(false);
  }, []);

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      return true;
    } catch (error) {
      console.error("Error accessing camera:", error);
      setAlertMessage("Camera access denied");
      return false;
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  // Draw stick figure on canvas
  const drawStickFigure = (ctx, keypoints, color = "#00ff00") => {
    const connections = [
      ["nose", "left_eye"], ["nose", "right_eye"],
      ["left_eye", "left_ear"], ["right_eye", "right_ear"],
      ["nose", "left_shoulder"], ["nose", "right_shoulder"],
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_elbow"], ["right_shoulder", "right_elbow"],
      ["left_elbow", "left_wrist"], ["right_elbow", "right_wrist"],
      ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
      ["left_hip", "right_hip"],
      ["left_hip", "left_knee"], ["right_hip", "right_knee"],
      ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
    ];

    const keypointMap = {};
    keypoints.forEach((kp) => {
      if (kp.score > 0.3) {
        keypointMap[kp.name] = kp;
      }
    });

    // Draw connections with glow effect
    ctx.shadowBlur = 15;
    ctx.shadowColor = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;

    connections.forEach(([start, end]) => {
      if (keypointMap[start] && keypointMap[end]) {
        ctx.beginPath();
        ctx.moveTo(keypointMap[start].x, keypointMap[start].y);
        ctx.lineTo(keypointMap[end].x, keypointMap[end].y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    ctx.fillStyle = color;
    Object.values(keypointMap).forEach((kp) => {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.shadowBlur = 0;
  };

  // Play alert sound
  const playAlertSound = () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = audioContextRef.current;
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    oscillator.frequency.value = 440;
    oscillator.type = "sine";
    gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
    
    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + 0.3);
  };

  // Main detection loop
  const detectPoses = useCallback(async () => {
    if (!detectorRef.current || !videoRef.current || !canvasRef.current) return;
    if (videoRef.current.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectPoses);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame with night vision effect
    ctx.filter = "brightness(1.2) contrast(1.3) saturate(0.5) hue-rotate(80deg)";
    ctx.drawImage(video, 0, 0);
    ctx.filter = "none";

    // Add scan lines effect
    ctx.fillStyle = "rgba(0, 0, 0, 0.03)";
    for (let i = 0; i < canvas.height; i += 4) {
      ctx.fillRect(0, i, canvas.width, 2);
    }

    try {
      const poses = await detectorRef.current.estimatePoses(video);
      setCurrentPoses(poses);

      if (poses.length > 0) {
        poses.forEach((pose) => {
          const validKeypoints = pose.keypoints.filter((kp) => kp.score > 0.3);
          if (validKeypoints.length > 5) {
            drawStickFigure(ctx, pose.keypoints, "#00ff00");
            
            // Update EMF based on detection confidence
            const avgScore = validKeypoints.reduce((a, b) => a + b.score, 0) / validKeypoints.length;
            const newEmf = Math.min(5, avgScore * 5 + Math.random() * 0.5);
            setEmfLevel(newEmf);

            // Alert on new detection
            const now = Date.now();
            if (now - lastAlertTime.current > 3000) {
              lastAlertTime.current = now;
              setDetectionCount((prev) => prev + 1);
              setAlertMessage("ENTITY DETECTED");
              playAlertSound();
              setTimeout(() => setAlertMessage(""), 2000);
            }
          }
        });
      } else {
        // Decay EMF when no detection
        setEmfLevel((prev) => Math.max(0, prev - 0.1));
      }
    } catch (error) {
      console.error("Detection error:", error);
    }

    animationRef.current = requestAnimationFrame(detectPoses);
  }, []);

  // Spirit Box frequency sweep
  useEffect(() => {
    if (!spiritBoxActive) return;

    const interval = setInterval(() => {
      setSpiritBoxFreq((prev) => {
        let next = prev + (Math.random() * 2 - 1) * 0.5;
        if (next > 108) next = 88;
        if (next < 88) next = 108;
        return Math.round(next * 10) / 10;
      });

      // Random chance to "receive" a word
      if (Math.random() < 0.05) {
        const word = SPIRIT_WORDS[Math.floor(Math.random() * SPIRIT_WORDS.length)];
        setSpiritWord(word);
        setTimeout(() => setSpiritWord(""), 1500);
      }
    }, 150);

    return () => clearInterval(interval);
  }, [spiritBoxActive]);

  // Start/Stop session
  const toggleSession = async () => {
    if (isRunning) {
      // Stop session
      cancelAnimationFrame(animationRef.current);
      stopCamera();
      setIsRunning(false);
      setSpiritBoxActive(false);
      
      // Update session end time
      if (sessionId) {
        try {
          await axios.put(`${API}/sessions/${sessionId}`, {
            session_end: new Date().toISOString(),
            total_detections: detectionCount,
            max_emf_level: emfLevel,
          });
        } catch (e) {
          console.error("Error updating session:", e);
        }
      }
    } else {
      // Start session
      if (!detectorRef.current) {
        await initializeDetector();
      }
      const cameraStarted = await startCamera();
      if (cameraStarted) {
        setIsRunning(true);
        setDetectionCount(0);
        setEmfLevel(0);
        detectPoses();
        
        // Create new session
        try {
          const res = await axios.post(`${API}/sessions`);
          setSessionId(res.data.id);
        } catch (e) {
          console.error("Error creating session:", e);
        }
      }
    }
  };

  // Take screenshot
  const takeScreenshot = async () => {
    if (!canvasRef.current) return;
    
    const imageData = canvasRef.current.toDataURL("image/png");
    
    try {
      await axios.post(`${API}/screenshots`, {
        image_data: imageData,
        detection_count: detectionCount,
        emf_level: emfLevel,
        notes: `Detections: ${detectionCount}, EMF: ${emfLevel.toFixed(1)}`,
      });
      setAlertMessage("SCREENSHOT SAVED");
      setTimeout(() => setAlertMessage(""), 2000);
    } catch (e) {
      console.error("Error saving screenshot:", e);
    }
  };

  // Log detection event
  const logDetection = async () => {
    if (currentPoses.length === 0) return;
    
    const pose = currentPoses[0];
    const validKeypoints = pose.keypoints.filter((kp) => kp.score > 0.3);
    const avgConfidence = validKeypoints.reduce((a, b) => a + b.score, 0) / validKeypoints.length;

    try {
      const res = await axios.post(`${API}/detections`, {
        detection_type: "pose",
        confidence: avgConfidence,
        keypoints_count: validKeypoints.length,
        emf_level: emfLevel,
        spirit_box_frequency: spiritBoxActive ? spiritBoxFreq : null,
      });
      
      setDetectionHistory((prev) => [res.data, ...prev].slice(0, 50));
      setAlertMessage("DETECTION LOGGED");
      setTimeout(() => setAlertMessage(""), 2000);
    } catch (e) {
      console.error("Error logging detection:", e);
    }
  };

  // Fetch detection history
  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API}/detections?limit=20`);
      setDetectionHistory(res.data);
    } catch (e) {
      console.error("Error fetching history:", e);
    }
  };

  useEffect(() => {
    initializeDetector();
    fetchHistory();
    return () => {
      cancelAnimationFrame(animationRef.current);
      stopCamera();
    };
  }, [initializeDetector]);

  // EMF needle rotation (0-5 scale mapped to -45 to 45 degrees)
  const emfNeedleRotation = -45 + (emfLevel / 5) * 90;
  
  // Spirit Box dial rotation
  const spiritDialRotation = ((spiritBoxFreq - 88) / 20) * 270 - 135;

  return (
    <div className="app-container" data-testid="ghost-camera-app">
      {/* Main Device Frame */}
      <div className="device-frame">
        {/* Corner Rivets */}
        <div className="rivet rivet-tl"></div>
        <div className="rivet rivet-tr"></div>
        <div className="rivet rivet-bl"></div>
        <div className="rivet rivet-br"></div>
        
        {/* Top Decorative Bar */}
        <div className="top-bar">
          <div className="status-light" style={{ background: isRunning ? '#00ff00' : '#333' }}></div>
          <span className="device-title">SPECTRAL LENS MK.IV</span>
          <div className="status-light" style={{ background: isRunning ? '#00ff00' : '#333' }}></div>
        </div>

        {/* Camera Viewport */}
        <div className="viewport-container">
          <div className="viewport-frame">
            <video
              ref={videoRef}
              style={{ display: "none" }}
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="viewport-canvas"
              data-testid="camera-canvas"
            />
            
            {!isRunning && (
              <div className="viewport-overlay">
                <span>{isLoading ? "INITIALIZING..." : "SYSTEM STANDBY"}</span>
              </div>
            )}
            
            {/* Alert Banner */}
            {alertMessage && (
              <div className="alert-banner" data-testid="alert-banner">
                {alertMessage}
              </div>
            )}
            
            {/* Detection Counter */}
            <div className="detection-counter">
              <span>ENTITIES: {detectionCount}</span>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          {/* Spirit Box Dial (Left) */}
          <div className="spirit-box-section">
            <div className="dial-container">
              <div className="dial-frame">
                <div className="dial-face">
                  <div className="dial-markings">
                    {[...Array(12)].map((_, i) => (
                      <div
                        key={i}
                        className="dial-mark"
                        style={{ transform: `rotate(${i * 30 - 135}deg)` }}
                      />
                    ))}
                  </div>
                  <div
                    className="dial-needle"
                    style={{ transform: `rotate(${spiritDialRotation}deg)` }}
                  />
                  <div className="dial-center"></div>
                </div>
              </div>
              <div className="dial-label">SPIRIT BOX</div>
              <div className="freq-display">
                {spiritBoxActive ? `${spiritBoxFreq.toFixed(1)} MHz` : "OFF"}
              </div>
              {spiritWord && (
                <div className="spirit-word" data-testid="spirit-word">
                  "{spiritWord}"
                </div>
              )}
            </div>
            
            {/* Pipe decoration */}
            <div className="pipe-connector"></div>
          </div>

          {/* Center Controls */}
          <div className="center-controls">
            {/* Small Gauges */}
            <div className="gauge-row">
              {/* EMF Meter */}
              <div className="emf-gauge">
                <div className="gauge-face">
                  <div className="gauge-scale">
                    {[0, 1, 2, 3, 4, 5].map((n) => (
                      <span key={n} className="gauge-number">{n}</span>
                    ))}
                  </div>
                  <div
                    className="gauge-needle"
                    style={{ transform: `rotate(${emfNeedleRotation}deg)` }}
                  />
                </div>
                <div className="gauge-label">EMF</div>
              </div>
            </div>
            
            {/* Control Buttons */}
            <div className="button-row">
              <button
                className={`brass-button main-button ${isRunning ? 'active' : ''}`}
                onClick={toggleSession}
                data-testid="toggle-session-btn"
              >
                <span>{isRunning ? "STOP" : "START"}</span>
              </button>
              
              <button
                className={`brass-button spirit-button ${spiritBoxActive ? 'active' : ''}`}
                onClick={() => setSpiritBoxActive(!spiritBoxActive)}
                data-testid="spirit-box-btn"
              >
                <span>SPIRIT</span>
              </button>
            </div>
            
            {/* Display Panel */}
            <div className="display-panel">
              <div className="display-screen" data-testid="display-screen">
                <span className="display-text">
                  {isRunning 
                    ? `EMF: ${emfLevel.toFixed(1)} | DET: ${detectionCount}`
                    : "READY"
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Right Panel */}
          <div className="right-panel">
            <button
              className="brass-button action-button"
              onClick={takeScreenshot}
              disabled={!isRunning}
              data-testid="screenshot-btn"
            >
              <span>📷</span>
            </button>
            
            <button
              className="brass-button action-button"
              onClick={logDetection}
              disabled={!isRunning || currentPoses.length === 0}
              data-testid="log-detection-btn"
            >
              <span>📝</span>
            </button>
            
            <button
              className="brass-button action-button"
              onClick={() => setShowHistory(!showHistory)}
              data-testid="history-btn"
            >
              <span>📜</span>
            </button>
          </div>
        </div>
        
        {/* Bottom Decorative Elements */}
        <div className="bottom-decoration">
          <div className="gear-icon">⚙</div>
          <div className="brand-plate">PARANORMAL INSTRUMENTS CO.</div>
          <div className="gear-icon">⚙</div>
        </div>
      </div>

      {/* History Modal */}
      {showHistory && (
        <div className="history-modal" data-testid="history-modal">
          <div className="history-content">
            <div className="history-header">
              <h2>DETECTION LOG</h2>
              <button onClick={() => setShowHistory(false)}>✕</button>
            </div>
            <div className="history-list">
              {detectionHistory.length === 0 ? (
                <p className="no-history">No detections recorded</p>
              ) : (
                detectionHistory.map((event, idx) => (
                  <div key={event.id || idx} className="history-item">
                    <div className="history-time">
                      {new Date(event.timestamp).toLocaleString()}
                    </div>
                    <div className="history-details">
                      <span>Type: {event.detection_type}</span>
                      <span>Confidence: {(event.confidence * 100).toFixed(0)}%</span>
                      <span>EMF: {event.emf_level?.toFixed(1) || "N/A"}</span>
                      <span>Keypoints: {event.keypoints_count}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
