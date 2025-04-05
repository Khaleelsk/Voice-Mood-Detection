
<body>

  <h1>🎙️ Voice Mood Detection</h1>
  <p>A machine learning-based project that detects the <strong>emotional mood</strong> of a person based on their voice input. The system processes voice recordings and classifies them into moods such as <em>happy</em>, <em>sad</em>, <em>angry</em>, <em>neutral</em>, etc.</p>

  <h2>🚀 Features</h2>
  <ul>
    <li>🎧 Accepts voice input (WAV/MP3)</li>
    <li>🔊 Extracts relevant audio features like MFCCs, Chroma, Spectral Contrast</li>
    <li>🧠 Uses trained ML models to predict emotion</li>
    <li>📊 Displays predicted mood with probability score</li>
    <li>💡 Streamlit web interface for real-time usage</li>
  </ul>

  <h2>🛠️ Tech Stack</h2>
  <ul>
    <li><strong>Frontend:</strong> Streamlit</li>
    <li><strong>Backend:</strong> Python, Librosa, NumPy, Pandas</li>
    <li><strong>Model:</strong> Logistic Regression / Random Forest / SVM</li>
    <li><strong>Libraries:</strong> scikit-learn, librosa, matplotlib, seaborn</li>
  </ul>

  <h2>📁 Project Structure</h2>
  <pre><code>
Voice-Mood-Detection/
├── VoiceMood-Detection.py                # Streamlit web app
├── model.pkl             # Trained ML model
├── Encoder.pkl             # Trained ML model
├── requirements.txt      # Dependencies
├── speech-emotion-recognition.ipynb     # Main ipynb
└── README.md             # Project documentation
  </code></pre>

  <h2>📦 Installation</h2>
  <p><strong>1. Clone the repo</strong></p>
  <pre><code>git clone https://github.com/Khaleelsk/Voice-Mood-Detection.git
cd Voice-Mood-Detection</code></pre>

  <p><strong>2. Install dependencies</strong></p>
  <pre><code>pip install -r requirements.txt</code></pre>

  <p><strong>3. Run the app</strong></p>
  <pre><code>streamlit run VoiceMood-Detection.py</code></pre>

  <h2>🎯 How It Works</h2>
  <ol>
    <li>Upload a <code>.wav</code> or <code>.mp3</code> voice file.</li>
    <li>The system extracts features like:
      <ul>
        <li>MFCC (Mel-frequency cepstral coefficients)</li>
        <li>Chroma</li>
        <li>Zero Crossing Rate</li>
      </ul>
    </li>
    <li>Features are passed to the trained classifier.</li>
    <li>The app predicts the emotion with a confidence score.</li>
  </ol>

  <h2>📊 Dataset</h2>
  <p>Trained using the <strong>RAVDESS</strong> (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset containing labeled audio samples.</p>

  <h2>🧠 Model Accuracy</h2>
  <p>Achieved approximately <strong>85% accuracy</strong> on test data. You can improve it by trying different models and tuning hyperparameters.</p>

  <h2>📌 Future Improvements</h2>
  <ul>
    <li>Real-time voice recording via microphone</li>
    <li>Support for multilingual emotion detection</li>
    <li>Deep learning model integration (LSTM/CNN)</li>
  </ul>

  <h2>🤝 Contributions</h2>
  <p>Feel free to open issues or pull requests to improve the project! UI tweaks, model upgrades, or bug fixes are all welcome.</p>

  <h2>📃 License</h2>
  <p>This project is licensed under the <strong>MIT License</strong>.</p>

</body>
</html>
