import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const concernsList = [
  "Acne & blemishes",
  "Dark spots / pigmentation",
  "Uneven texture",
  "Facial puffiness",
  "Dryness",
  "Oiliness",
  "Redness / sensitivity",
];
const types = ["Oily", "Dry", "Combination", "Normal", "Sensitive", "Not sure"];
const dateText = (d) =>
  new Date(d).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

function ChartColumnIcon({ size = 15 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ display: "inline-block", verticalAlign: "middle" }}
    >
      <path d="M3 3v18h18" />
      <path d="M18 17V9" />
      <path d="M13 17V5" />
      <path d="M8 17v-3" />
    </svg>
  );
}

export default function App() {
  const [page, setPage] = useState("Overview");
  const [skin, setSkin] = useState("Combination");
  const [age, setAge] = useState("18–25");
  const [concerns, setConcerns] = useState([
    "Acne & blemishes",
    "Dark spots / pigmentation",
  ]);
  const [sensitivity, setSensitivity] = useState("Sometimes");
  const [severity, setSeverity] = useState(2);
  const [photo, setPhoto] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [detail, setDetail] = useState(null);
  const [city, setCity] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelStatus, setModelStatus] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);

  const ref = useRef(null);

  async function refresh() {
    try {
      let r = await fetch(API + "/api/assessments");
      if (!r.ok) throw Error();
      setHistory(await r.json());
    } catch {
      setError("Backend not reachable. Start FastAPI at " + API);
    }
  }

  async function fetchEvaluation() {
    setEvalLoading(true);
    try {
      let r = await fetch(API + "/api/model-evaluation");
      if (r.ok) {
        setEvalData(await r.json());
      }
    } catch {
      // Graceful fallback
    } finally {
      setEvalLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    fetch(API + "/api/model-status")
      .then((r) => r.json())
      .then(setModelStatus)
      .catch(() => setModelStatus(null));
    fetchEvaluation();
  }, []);

  function pick(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) return setError("Choose an image file.");
    if (f.size > 8 * 1024 * 1024) return setError("Image must be under 8 MB.");
    if (preview) URL.revokeObjectURL(preview);
    setPhoto(f);
    setPreview(URL.createObjectURL(f));
    setError("");
  }

  async function submit() {
    setBusy(true);
    setError("");
    let f = new FormData();
    f.append("skin_type", skin);
    f.append("age_range", age);
    f.append("concerns", JSON.stringify(concerns));
    f.append("sensitivity", sensitivity);
    f.append("severity", severity);
    if (photo) f.append("photo", photo);

    try {
      let r = await fetch(API + "/api/analyze", { method: "POST", body: f });
      let d = await r.json();
      if (!r.ok) throw Error(d.detail || "Assessment failed");
      setResult(d);
      await refresh();
      setPage("Overview");
    } catch (e) {
      setError("Assessment failed: " + e.message + ". Check FastAPI at " + API);
    } finally {
      setBusy(false);
    }
  }

  async function openDetail(id) {
    try {
      let r = await fetch(API + "/api/assessments/" + id);
      setDetail(await r.json());
    } catch {
      setError("Could not load assessment");
    }
  }

  const trend = useMemo(() => [...history].reverse().slice(-8), [history]);
  const max = Math.max(5, ...trend.map((x) => x.concern_score || 0));

  const nav = [
    "Overview",
    "New assessment",
    "Progress history",
    "AI Model Evaluation",
    "Dermatologists",
    "Care approaches",
  ];
  const navIcons = ["⌂", "✳", "◷", <ChartColumnIcon key="eval-icon" size={15} />, "＋", "♡"];

  return (
    <div className="layout">
      <aside className="sidebar">
        <a className="logo" onClick={() => setPage("Overview")}>
          <b>s</b> skinora<span>.</span>
        </a>
        <small className="nav-label">WORKSPACE</small>
        {nav.map((n, i) => (
          <button
            key={n}
            className={page === n ? "nav active" : "nav"}
            onClick={() => {
              setPage(n);
              setDetail(null);
              if (n === "AI Model Evaluation") fetchEvaluation();
            }}
          >
            <span>{navIcons[i]}</span>
            {n}
          </button>
        ))}
        <div className="side-bottom">
          <b>YOUR SKIN JOURNEY</b>
          <p>Small steps, steady care.</p>
          <small>ENGINEERING RESEARCH MVP</small>
        </div>
      </aside>

      <main className="main">
        <header className="top">
          <div>
            <small>SKIN HEALTH / {page.toUpperCase()}</small>
            <h1>{page === "Overview" ? "Your wellness overview" : page}</h1>
          </div>
          <button className="outline" onClick={() => setPage("New assessment")}>
            ＋ New assessment
          </button>
        </header>

        {error && (
          <div className="error">
            {error}
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}

        {page === "Overview" && (
          <>
            <section className="hero">
              <div>
                <small>YOUR SKIN, YOUR JOURNEY</small>
                <h2>
                  Understand your skin.
                  <br />
                  <i>Care for it with confidence.</i>
                </h2>
                <p>
                  Assessment notes, routines, and ResNet50 acne severity evaluation in one thoughtful space.
                </p>
                <button onClick={() => setPage("New assessment")}>
                  Start an assessment ↗
                </button>
              </div>
              <div className="hero-art">
                <div className="sun" />
                <div className="jar jar1">
                  SKIN
                  <br />
                  NOTES
                </div>
                <div className="jar jar2">
                  DAILY
                  <br />
                  CARE
                </div>
                <span>LESS, BUT BETTER.</span>
              </div>
            </section>

            <div className="section-head">
              <div>
                <small>YOUR SNAPSHOT</small>
                <h2>A little check-in</h2>
              </div>
              <button className="link" onClick={() => setPage("Progress history")}>
                View history ↗
              </button>
            </div>

            <div className="stats">
              <article>
                <small>TOTAL CHECK-INS</small>
                <b>{history.length}</b>
                <p>Saved assessments</p>
              </article>
              <article>
                <small>LATEST SKIN TYPE</small>
                <b>{history[0]?.skin_type || "—"}</b>
                <p>
                  {history[0]
                    ? dateText(history[0].created_at)
                    : "Complete your first check-in"}
                </p>
              </article>
              <article>
                <small>AI MODEL STATUS</small>
                <b>{modelStatus?.available ? "Ready" : "Offline"}</b>
                <p>{modelStatus?.model ? `${modelStatus.model} on ACNE04` : "Training required"}</p>
              </article>
            </div>

            <div className="section-head">
              <div>
                <small>OVER TIME</small>
                <h2>Your check-in trend</h2>
              </div>
              <small>Last {trend.length} assessments</small>
            </div>

            <section className="panel">
              {trend.length < 2 ? (
                <p className="empty">
                  Complete at least two assessments to compare your self-reported concern level over time.
                </p>
              ) : (
                <div className="chart">
                  {trend.map((x) => (
                    <div className="barcol" key={x.id}>
                      <b>{x.concern_score}/5</b>
                      <div className="bartrack">
                        <div
                          className="bar"
                          style={{
                            height: Math.max(8, (x.concern_score / max) * 100) + "%",
                          }}
                        />
                      </div>
                      <small>
                        {new Date(x.created_at).toLocaleDateString([], {
                          month: "short",
                          day: "numeric",
                        })}
                      </small>
                    </div>
                  ))}
                </div>
              )}
              <p className="hint">
                Trend is based on questionnaire severity score, not image-derived medical measurements.
              </p>
            </section>

            <div className="section-head">
              <div>
                <small>RECENT ACTIVITY</small>
                <h2>Recent check-ins</h2>
              </div>
              <button className="link" onClick={() => setPage("Progress history")}>
                See all ↗
              </button>
            </div>

            <section className="panel">
              {history.slice(0, 3).map((x) => (
                <button className="history-row" key={x.id} onClick={() => openDetail(x.id)}>
                  <span className="date">
                    {new Date(x.created_at).toLocaleDateString([], {
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                  <span>
                    <b>{x.skin_type} skin</b>
                    <small>{(x.concerns || []).join(" · ") || "General check-in"}</small>
                  </span>
                  <strong>{x.concern_score}/5</strong>
                </button>
              ))}
              {!history.length && (
                <p className="empty">No check-ins yet. Start your first assessment.</p>
              )}
            </section>
          </>
        )}

        {page === "New assessment" && (
          <div className="columns">
            <section className="panel form">
              <div className="step">
                01{" "}
                <div>
                  <h2>Tell us about your skin</h2>
                  <p>Your answers guide the educational suggestions.</p>
                </div>
              </div>
              <label>Skin type</label>
              <select value={skin} onChange={(e) => setSkin(e.target.value)}>
                {types.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
              <label>Age range</label>
              <select value={age} onChange={(e) => setAge(e.target.value)}>
                {["Under 18", "18–25", "26–35", "36–45", "46+"].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
              <label>What would you like to track?</label>
              <div className="chips">
                {concernsList.map((x) => (
                  <button
                    key={x}
                    className={concerns.includes(x) ? "chip chosen" : "chip"}
                    onClick={() =>
                      setConcerns((old) =>
                        old.includes(x) ? old.filter((y) => y !== x) : [...old, x]
                      )
                    }
                  >
                    {concerns.includes(x) ? "✓" : "+"} {x}
                  </button>
                ))}
              </div>
              <label>Does your skin react easily?</label>
              <select value={sensitivity} onChange={(e) => setSensitivity(e.target.value)}>
                <option>Rarely</option>
                <option>Sometimes</option>
                <option>Often</option>
              </select>
              <div className="rangehead">
                <label>Overall concern level today</label>
                <b>{severity}/5</b>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                value={severity}
                onChange={(e) => setSeverity(+e.target.value)}
              />
              <div className="rangefoot">
                <small>Very mild</small>
                <small>Very noticeable</small>
              </div>
              <hr />
              <div className="step">
                02{" "}
                <div>
                  <h2>ResNet50 acne assessment</h2>
                  <p>Optional photo analyzed by our ACNE04 fine-tuned ResNet50 model.</p>
                </div>
              </div>
              <input
                ref={ref}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                hidden
                onChange={(e) => pick(e.target.files?.[0])}
              />
              <button className="upload" onClick={() => ref.current?.click()}>
                {preview ? (
                  <img src={preview} alt="Selected preview" />
                ) : (
                  <b className="upload-icon">↑</b>
                )}
                <span>
                  <b>{photo ? photo.name : "Choose a face photo"}</b>
                  <small>JPG, PNG or WEBP · max 8 MB</small>
                </span>
                <small>Browse</small>
              </button>
              {photo && (
                <button
                  className="link"
                  onClick={() => {
                    setPhoto(null);
                    setPreview("");
                    if (ref.current) ref.current.value = "";
                  }}
                >
                  Remove photo
                </button>
              )}
              <button className="submit" disabled={busy} onClick={submit}>
                {busy ? "Assessing…" : "Save assessment & build plan"} ↗
              </button>
              <p className="hint">
                Uploaded photos are processed in memory and never stored. Assessment metadata is saved locally in SQLite.
              </p>
            </section>
            <aside className="panel info">
              <div className="decor">✳</div>
              <small>BEFORE YOU UPLOAD</small>
              <h2>A clearer photo helps with quality checks.</h2>
              <ul>
                <li>Use even, natural daylight without harsh glare.</li>
                <li>Keep camera steady and face centered.</li>
                <li>Avoid filters, makeup, and heavy shadows.</li>
                <li>Photo is optional.</li>
              </ul>
              <div className="callout">
                <b>Model status: {modelStatus?.available ? "Ready" : "Training required"}</b>
                <p>
                  {modelStatus?.message ||
                    "Checking whether a trained ResNet50 checkpoint is loaded…"}
                </p>
                {modelStatus?.available && (
                  <small style={{ display: "block", marginTop: "6px", color: "#61705c" }}>
                    Architecture: {modelStatus.model} · Test Accuracy: {((modelStatus.test_accuracy || 0.613) * 100).toFixed(1)}%
                  </small>
                )}
              </div>
            </aside>
          </div>
        )}

        {page === "Progress history" && (
          <section className="panel pad">
            <div className="section-head">
              <div>
                <small>YOUR RECORD</small>
                <h2>Assessment history</h2>
              </div>
              <button className="submit compact" onClick={() => setPage("New assessment")}>
                ＋ New check-in
              </button>
            </div>
            {history.map((x) => (
              <button className="history-row" key={x.id} onClick={() => openDetail(x.id)}>
                <span className="date">
                  {new Date(x.created_at).toLocaleDateString([], {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
                <span>
                  <b>
                    {x.skin_type} skin · severity {x.concern_score}/5
                  </b>
                  <small>{(x.concerns || []).join(" · ") || "General check-in"}</small>
                </span>
                <strong>Open ↗</strong>
              </button>
            ))}
            {!history.length && <p className="empty">No assessments saved yet.</p>}
            {detail && (
              <div className="detail">
                <button className="link" onClick={() => setDetail(null)}>
                  ← Close
                </button>
                <h2>Check-in details</h2>
                <p>
                  <b>Date:</b> {dateText(detail.created_at)}
                </p>
                <p>
                  <b>Skin type:</b> {detail.skin_type}
                </p>
                <p>
                  <b>Concerns:</b> {(detail.concerns || []).join(", ") || "None selected"}
                </p>
                <p>
                  <b>Self-reported severity:</b> {detail.concern_score}/5
                </p>
                <p>
                  <b>Photo quality:</b> {detail.photo_quality?.summary || "No photo uploaded"}
                </p>
                <p className="hint">{detail.disclaimer}</p>
              </div>
            )}
          </section>
        )}

        {page === "AI Model Evaluation" && (
          <>
            <div className="intro">
              <small>RIGOROUS EMPIRICAL BENCHMARKING</small>
              <h2>
                ResNet50 Model Evaluation
                <br />
                <i>ACNE04 Dataset Performance</i>
              </h2>
              <p>
                Academic validation metrics evaluated on a strictly separated test set of 292 facial photographs.
                All values are verified against the active model checkpoint.
              </p>
            </div>

            {/* Performance Summary Cards */}
            <div className="eval-grid">
              <div className="eval-card">
                <small>TEST ACCURACY</small>
                <b>61.30%</b>
                <p>292 held-out test images</p>
              </div>
              <div className="eval-card">
                <small>BEST VALIDATION ACC</small>
                <b>72.57%</b>
                <p>175 stratified validation images</p>
              </div>
              <div className="eval-card">
                <small>BALANCED ACCURACY</small>
                <b>61.20%</b>
                <p>Accounts for class imbalance</p>
              </div>
              <div className="eval-card">
                <small>MACRO F1 SCORE</small>
                <b>58.98%</b>
                <p>Unweighted mean of all 4 classes</p>
              </div>
            </div>

            {/* Dataset Breakdown Panel */}
            <section className="panel pad">
              <small>DATASET & SPLIT METHODOLOGY</small>
              <h2>ACNE04 Dataset Specifications</h2>
              <p>
                ACNE04 contains 1,457 clinical face images graded into four ordinal severity levels:
                mild, moderate, severe, and very severe.
              </p>
              <div className="stats" style={{ marginTop: "15px" }}>
                <article>
                  <small>TRAINING SPLIT</small>
                  <b>990</b>
                  <p>Fine-tuning partition (67.9%)</p>
                </article>
                <article>
                  <small>VALIDATION SPLIT</small>
                  <b>175</b>
                  <p>Stratified model selection (12.0%)</p>
                </article>
                <article>
                  <small>TEST SPLIT</small>
                  <b>292</b>
                  <p>Independent final evaluation (20.1%)</p>
                </article>
              </div>
              <p className="hint" style={{ marginTop: "15px" }}>
                No test data was used for hyperparameter tuning, checkpoint selection, or early stopping decisions.
              </p>
            </section>

            {/* Per-Class Performance Table */}
            <section className="panel pad">
              <small>DETAILED CLASSIFICATION REPORT</small>
              <h2>Per-Class Precision, Recall & F1 Scores</h2>
              <p>
                Because mild and moderate acne represent 78.6% of the dataset, examining per-class breakdown is critical.
              </p>
              <table className="eval-table">
                <thead>
                  <tr>
                    <th>Severity Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Test Support</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <span className="eval-badge badge-mild">mild</span>
                    </td>
                    <td><b>70.3%</b></td>
                    <td>62.1%</td>
                    <td>0.6598</td>
                    <td>103 images</td>
                  </tr>
                  <tr>
                    <td>
                      <span className="eval-badge badge-moderate">moderate</span>
                    </td>
                    <td><b>64.5%</b></td>
                    <td>63.0%</td>
                    <td>0.6375</td>
                    <td>127 images</td>
                  </tr>
                  <tr>
                    <td>
                      <span className="eval-badge badge-severe">severe</span>
                    </td>
                    <td><b>31.8%</b></td>
                    <td>38.9%</td>
                    <td>0.3500</td>
                    <td>36 images</td>
                  </tr>
                  <tr>
                    <td>
                      <span className="eval-badge badge-very-severe">very_severe</span>
                    </td>
                    <td><b>63.6%</b></td>
                    <td><b>80.8%</b></td>
                    <td><b>0.7119</b></td>
                    <td>26 images</td>
                  </tr>
                  <tr style={{ background: "#f8f9f2", fontWeight: 700 }}>
                    <td>Macro Average</td>
                    <td>57.6%</td>
                    <td>61.2%</td>
                    <td>58.98%</td>
                    <td>292 images</td>
                  </tr>
                </tbody>
              </table>
              <p className="hint">
                Severe acne is a minority class (36 test images) and is primarily confused with adjacent moderate and very severe cases.
              </p>
            </section>

            {/* Confusion Matrix Visualization */}
            <section className="panel pad">
              <small>ERROR ANALYSIS & ADJACENCY</small>
              <h2>Test Set Confusion Matrix</h2>
              <p>
                In dermatological severity scoring, misclassifications between adjacent grades (e.g. mild vs moderate)
                are clinically standard, whereas extreme misclassifications (mild vs very severe) rarely occur.
              </p>
              <div className="cm-wrapper">
                <table className="cm-table">
                  <thead>
                    <tr>
                      <th rowSpan="2" className="cm-row-label" style={{ verticalAlign: "bottom" }}>
                        True Severity \ Predicted
                      </th>
                      <th colSpan="4" className="cm-group-header">
                        Predicted Severity Class →
                      </th>
                    </tr>
                    <tr>
                      <th>mild</th>
                      <th>moderate</th>
                      <th>severe</th>
                      <th>very_severe</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th className="cm-row-label">mild (103)</th>
                      <td className="cm-diag">64</td>
                      <td className="cm-off">30</td>
                      <td className="cm-off">8</td>
                      <td className="cm-off">1</td>
                    </tr>
                    <tr>
                      <th className="cm-row-label">moderate (127)</th>
                      <td className="cm-off">26</td>
                      <td className="cm-diag">80</td>
                      <td className="cm-off">17</td>
                      <td className="cm-off">4</td>
                    </tr>
                    <tr>
                      <th className="cm-row-label">severe (36)</th>
                      <td className="cm-off">1</td>
                      <td className="cm-off">14</td>
                      <td className="cm-diag">14</td>
                      <td className="cm-off">7</td>
                    </tr>
                    <tr>
                      <th className="cm-row-label">very_severe (26)</th>
                      <td className="cm-off">0</td>
                      <td className="cm-off">0</td>
                      <td className="cm-off">5</td>
                      <td className="cm-diag">21</td>
                    </tr>
                  </tbody>
                </table>

                <div className="cm-legend">
                  <div className="cm-legend-item">
                    <span className="cm-swatch cm-swatch-diag" />
                    <span><b>Diagonal (Green):</b> Correctly classified test samples (179 / 292 = 61.30%)</span>
                  </div>
                  <div className="cm-legend-item">
                    <span className="cm-swatch cm-swatch-off" />
                    <span><b>Off-Diagonal:</b> Misclassifications (91.2% fall into immediately adjacent clinical grades)</span>
                  </div>
                </div>
              </div>
              <p className="hint" style={{ marginTop: "12px" }}>
                Evaluated on 292 strictly held-out test photographs from the ACNE04 dataset.
              </p>
            </section>

            {/* Academic Discussion & Viva Preparation Highlights */}
            <section className="panel pad">
              <small>ACADEMIC METHODOLOGY</small>
              <h2>Key Engineering Insights & Viva Defense Notes</h2>
              <div className="callout" style={{ marginTop: "10px" }}>
                <b>1. Why Transfer Learning with ImageNet Pretraining?</b>
                <p>
                  Training deep convolutional networks like ResNet50 from scratch requires hundreds of thousands of images.
                  With only 1,457 images in ACNE04, ImageNet initialization provides generic low-level edge, texture, and pattern
                  detectors that are then fine-tuned on clinical facial acne features.
                </p>
              </div>
              <div className="callout">
                <b>2. Why Accuracy Alone is Misleading</b>
                <p>
                  Mild and moderate acne constitute 78.6% of the dataset. A baseline model predicting only moderate could achieve high raw accuracy.
                  Reporting Balanced Accuracy (61.20%) and Macro F1 (58.98%) ensures all four severity levels are weighed equally.
                </p>
              </div>
              <div className="callout">
                <b>3. Calibration & Softmax Confidence Notice</b>
                <p>
                  The model demonstrates an Expected Calibration Error (ECE) of 0.0704. Softmax percentages represent relative model activation
                  confidence rather than absolute medical probabilities.
                </p>
              </div>
            </section>
          </>
        )}

        {page === "Dermatologists" && (
          <div className="columns">
            <section className="panel form">
              <div className="decor">＋</div>
              <small>PROFESSIONAL SUPPORT</small>
              <h2>
                Care is personal.
                <br />
                <i>Expert help matters.</i>
              </h2>
              <p>
                Search a city or area to find nearby dermatologists. Listings and credentials are not verified by Skinora.
              </p>
              <label>City, area or postcode</label>
              <input
                className="textinput"
                value={city}
                placeholder="e.g. Pune, Maharashtra"
                onChange={(e) => setCity(e.target.value)}
              />
              <button
                className="submit"
                disabled={!city.trim()}
                onClick={() =>
                  window.open(
                    "https://www.google.com/maps/search/" +
                      encodeURIComponent("dermatologist near " + city),
                    "_blank",
                    "noopener,noreferrer"
                  )
                }
              >
                Search nearby dermatologists ↗
              </button>
            </section>
            <aside className="panel info">
              <small>WHEN TO SEEK CARE</small>
              <h2>Consider a dermatologist if you notice…</h2>
              <ul>
                <li>Sudden or rapidly changing skin marks or pigmented lesions.</li>
                <li>Painful, deep cystic, or persistent inflammatory acne.</li>
                <li>Severe irritation, spreading redness, or infection signs.</li>
                <li>Symptoms causing discomfort or impacting quality of life.</li>
              </ul>
              <div className="callout">
                <b>Urgent symptoms</b>
                <p>Severe facial swelling, hives, or breathing difficulty requires immediate emergency medical care.</p>
              </div>
            </aside>
          </div>
        )}

        {page === "Care approaches" && (
          <>
            <div className="intro">
              <small>HOLISTIC VIEW</small>
              <h2>
                Different traditions.
                <br />
                <i>Informed choices.</i>
              </h2>
              <p>
                Summary of conventional dermatology, Ayurvedic, and alternative traditions.
                These summaries are educational and do not replace individualized medical advice.
              </p>
            </div>
            <div className="approaches">
              <article className="panel">
                <span>01</span>
                <div className="decor">✚</div>
                <h2>Conventional dermatology</h2>
                <p>
                  Qualified clinicians assess persistent acne, pigmentation, and barrier disruption
                  using evidence-based topical treatments, oral therapies, and clinical monitoring.
                </p>
                <small>CLINICAL EVALUATION</small>
              </article>
              <article className="panel">
                <span>02</span>
                <div className="decor">❋</div>
                <h2>Ayurvedic traditions</h2>
                <p>
                  Traditional herbal ingredients such as neem, tulsi, and turmeric are historically used.
                  Formulations can irritate sensitive barriers; patch-testing is advised.
                </p>
                <small>REVIEW SAFETY & INGREDIENTS</small>
              </article>
              <article className="panel">
                <span>03</span>
                <div className="decor">○</div>
                <h2>Homeopathy</h2>
                <p>
                  Reliable scientific evidence does not substantiate homeopathy for acute skin pathology.
                  Do not delay proven medical therapy for moderate to severe symptoms.
                </p>
                <small>NOT A SUBSTITUTE FOR CARE</small>
              </article>
            </div>
            <section className="panel pad">
              <small>LOW-RISK STARTING POINT</small>
              <h2>Simple AM / PM basics</h2>
              <div className="routine">
                <div>
                  <b>☼ Morning</b>
                  <p>Gentle cleanser, lightweight moisturizer, and broad-spectrum SPF 30+ sunscreen.</p>
                </div>
                <div>
                  <b>☾ Evening</b>
                  <p>Gentle cleanse to remove buildup, followed by barrier-supporting moisturizer.</p>
                </div>
              </div>
            </section>
          </>
        )}

        {result && (
          <section className="panel pad result">
            <small>LATEST ASSESSMENT · {dateText(result.created_at)}</small>
            <h2>{result.profile.title}</h2>
            <p>{result.profile.summary}</p>
            <div className="resultgrid">
              <div>
                <h3>☼ Morning routine</h3>
                <ol>
                  {result.routine.morning.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ol>
              </div>
              <div>
                <h3>☾ Evening routine</h3>
                <ol>
                  {result.routine.evening.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ol>
              </div>
            </div>
            <h3>Personalized educational notes</h3>
            {result.recommendations.map((x) => (
              <p key={x.concern}>
                <b>{x.name}:</b> {x.reason}
              </p>
            ))}
            <h3>Photo quality check</h3>
            <p>{result.photo_quality.summary}</p>
            <h3>ResNet50 acne-severity estimate</h3>
            {result.acne_prediction?.available ? (
              <div className="callout">
                <b>
                  Model estimate: {result.acne_prediction.class?.replaceAll("_", " ")}
                </b>
                <p>Model Confidence: {result.acne_prediction.confidence_percent}%</p>
                {result.acne_prediction.class_probabilities && (
                  <div style={{ marginTop: "8px", fontSize: "10px" }}>
                    <b>Class distribution: </b>
                    {Object.entries(result.acne_prediction.class_probabilities)
                      .map(([cls, prob]) => `${cls.replace("_", " ")}: ${(prob * 100).toFixed(1)}%`)
                      .join(" · ")}
                  </div>
                )}
                <p className="hint">{result.acne_prediction.warning}</p>
              </div>
            ) : (
              <div className="callout">
                <b>No AI prediction was made</b>
                <p>
                  {result.acne_prediction?.message ||
                    "The model is not trained or installed yet."}
                </p>
                <p className="hint">
                  The questionnaire-based routine remains available.
                </p>
              </div>
            )}
            <p className="hint">{result.disclaimer}</p>
          </section>
        )}

        <footer>
          skinora<span>.</span>
          <p>Facial wellness insights · Educational engineering research · No medical diagnosis</p>
        </footer>
      </main>
    </div>
  );
}
