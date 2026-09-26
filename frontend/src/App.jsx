import { useEffect, useRef, useState } from "react";
import "./App.css";
import AvatarScene from "./AvatarScene";
function App() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const recognitionTimerRef = useRef(null);
  const predictionHistoryRef = useRef([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");

  const [prediction, setPrediction] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [handsDetected, setHandsDetected] = useState(0);
  const [recognizing, setRecognizing] = useState(false);
  const [recognitionMessage, setRecognitionMessage] = useState(
    "Waiting for recognition"
  );

  const recognizeFrame = async () => {
  if (!videoRef.current || !canvasRef.current) {
    return;
  }

  const video = videoRef.current;

  if (
    video.readyState < 2 ||
    video.videoWidth === 0 ||
    video.videoHeight === 0
  ) {
    return;
  }

  const canvas = canvasRef.current;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext("2d");

  context.drawImage(
    video,
    0,
    0,
    canvas.width,
    canvas.height
  );

  canvas.toBlob(
    async (blob) => {
      if (!blob) {
        return;
      }

      try {
        setRecognizing(true);

        const formData = new FormData();

        formData.append(
          "file",
          blob,
          "camera-frame.jpg"
        );

        const response = await fetch(
          "http://127.0.0.1:8000/predict",
          {
            method: "POST",
            body: formData
          }
        );

        if (!response.ok) {
          throw new Error(
            `Server error: ${response.status}`
          );
        }

        const data = await response.json();

        if (data.success) {
          const history = predictionHistoryRef.current;

          history.push({
            letter: data.prediction,
            confidence: data.confidence,
          });

          if (history.length > 5) {
            history.shift();
          }

          const counts = {};

          history.forEach((item) => {
            counts[item.letter] =
              (counts[item.letter] || 0) + 1;
          });

          const stableLetter = Object.keys(counts).reduce(
            (a, b) =>
              counts[a] >= counts[b] ? a : b
          );

          const stablePredictions = history.filter(
            (item) => item.letter === stableLetter
          );

          const averageConfidence =
            stablePredictions.reduce(
              (sum, item) => sum + item.confidence,
              0
            ) / stablePredictions.length;

          setPrediction(stableLetter);
          setConfidence(Number(averageConfidence.toFixed(2)));
          setHandsDetected(data.hands_detected);

          setRecognitionMessage(
            averageConfidence >= 70
              ? "High confidence"
              : "Low confidence"
          );
        }
        else {
          setPrediction("-");
          setConfidence(0);
          setHandsDetected(0);

          setRecognitionMessage(
            data.message || "No hand detected"
          );
        }

      } catch (error) {
        console.error(
          "Recognition error:",
          error
        );

        setRecognitionMessage(
          "Unable to connect to AI server"
        );

      } finally {
        setRecognizing(false);
      }
    },
    "image/jpeg",
    0.8
  );
};

  const startCamera = async () => {
    try {
      setCameraError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraOn(true);

      setTimeout(() => {
        recognizeFrame();
        
        recognitionTimerRef.current = setInterval(
          recognizeFrame,700);
        }, 1000);

    } catch (error) {
      console.error(error);
      setCameraError(
        "Unable to access camera. Please allow camera permission."
      );
    }
  };
  
  const stopCamera = () => {
    if (recognitionTimerRef.current) {
      clearInterval(
        recognitionTimerRef.current
      );

      recognitionTimerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (videoRef.current) {
     videoRef.current.srcObject = null;
    }
    
    setCameraOn(false);

    setPrediction("-");
    setConfidence(0);
    setHandsDetected(0);
    predictionHistoryRef.current = [];
    setRecognitionMessage(
      "Waiting for recognition"
    );
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  return (
    <div className="app">

      <canvas
      ref={canvasRef}
      style={{ display: "none" }}
      />

      <header className="topbar">

        <div className="brand">

          <div className="brand-icon">
            ✦
          </div>

          <div>
            <h1>SignFlow AI</h1>
            <p>Indian Sign Language Recognition</p>
          </div>

        </div>

        <div className="system-status">

          <span className="status-dot"></span>

          {cameraOn ? "Camera Active" : "System Online"}

        </div>

      </header>


      <main className="dashboard">

        <div className="welcome">

          <div>

            <span className="eyebrow">
              AI SIGN LANGUAGE ASSISTANT
            </span>

            <h2>
              Translate signs into
              <span> meaningful communication.</span>
            </h2>

            <p>
              Show an Indian Sign Language alphabet sign to the camera
              and let the AI recognize it in real time.
            </p>

          </div>

          <div className="welcome-badge">
            <span>●</span>
            ISL A–Z
          </div>

        </div>


        <div className="workspace">


          {/* CAMERA */}

          <section className="panel camera-panel">

            <div className="panel-header">

              <div>

                <span className="panel-label">
                  SIGN INPUT
                </span>

                <h3>
                  Camera
                </h3>

              </div>


              <span className="live-badge">

                <span></span>

                {cameraOn ? "LIVE" : "READY"}

              </span>

            </div>


            <div className="camera-view">

              {!cameraOn && (
                <>
                  <div className="corner top-left"></div>
                  <div className="corner top-right"></div>
                  <div className="corner bottom-left"></div>
                  <div className="corner bottom-right"></div>

                  <div className="camera-placeholder">

                    <div className="camera-icon">
                      ⌾
                    </div>

                    <h4>
                      Camera Preview
                    </h4>

                    <p>
                      Start recognition to activate your camera
                    </p>

                  </div>
                </>
              )}


              <video
                ref={videoRef}
                className={`camera-video ${
                  cameraOn ? "visible" : ""
                }`}
                autoPlay
                playsInline
                muted
              />


              {cameraOn && (
                <div className="camera-overlay">

                  <div className="corner top-left"></div>
                  <div className="corner top-right"></div>
                  <div className="corner bottom-left"></div>
                  <div className="corner bottom-right"></div>

                  <div className="camera-hint">
                    Position your hand inside the frame
                  </div>

                </div>
              )}

            </div>


            {cameraError && (
              <div className="camera-error">
                {cameraError}
              </div>
            )}


            {!cameraOn ? (

              <button
                className="primary-button"
                onClick={startCamera}
              >
                <span>▶</span>
                Start Recognition
              </button>

            ) : (

              <button
                className="stop-button"
                onClick={stopCamera}
              >
                <span>■</span>
                Stop Camera
              </button>

            )}

          </section>


          {/* AVATAR */}

          <section className="panel avatar-panel">

            <div className="panel-header">

              <div>

                <span className="panel-label">
                  SIGN REPRESENTATION
                </span>

                <h3>
                  3D Avatar
                </h3>

              </div>

              <span className="ready-badge">
                Ready
              </span>

            </div>


            <div className="avatar-view">

              <div className="avatar-glow"></div>

              <AvatarScene />

            </div>


            <div className="avatar-info">

              <span>
                Current representation
              </span>

              <strong>
                Waiting for sign...
              </strong>

            </div>

          </section>

        </div>


        {/* RESULTS */}

        <div className="results">


          <div className="result-card prediction-card">

            <div className="result-heading">

              <span className="result-icon">
                ✦
              </span>

              <div>

                <span>
                  DETECTED SIGN
                </span>

                <small>
                  AI prediction
                </small>

              </div>

            </div>

            <div className="prediction">
              <span>{prediction}</span>
            </div>

          </div>


          <div className="result-card confidence-card">

            <div className="result-heading">

              <span className="result-icon">
                ◉
              </span>

              <div>

                <span>
                  CONFIDENCE
                </span>

                <small>
                  Recognition accuracy
                </small>

              </div>

            </div>

            <div className="confidence-value">
              {confidence > 0 ? `${confidence}%` : "—"}
            </div>

            <div className="progress">

              <div
                className="progress-fill"
                style={{ width: `${confidence}%` }}
              ></div>

            </div>

            <div className="confidence-text">
              {recognizing
              ? "Analyzing..."
              : recognitionMessage}
            </div>

          </div>


          <div className="result-card hands-card">

            <div className="result-heading">

              <span className="result-icon">
                ♧
              </span>

              <div>

                <span>
                  HANDS DETECTED
                </span>

                <small>
                  MediaPipe tracking
                </small>

              </div>

            </div>

            <div className="hands-value">
              {handsDetected || "—"}
            </div>

            <div className="hands-text">
              {handsDetected > 0
              ? "Hands detected"
              : "Waiting for hand"}
            </div>

          </div>

        </div>

      </main>


      <footer>

        <span>
          SignFlow AI
        </span>

        <span>
          AI-powered Indian Sign Language Recognition
        </span>

        <span>
          ISL Alphabet • A–Z
        </span>

      </footer>

    </div>
  );
}

export default App;