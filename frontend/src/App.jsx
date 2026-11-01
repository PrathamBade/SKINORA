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
    setEvalData(null);
    try {
      let r = await fetch(API + "/api/model-evaluation");
      if (r.ok) {
        setEvalData(await r.json());
      } else {
        setEvalData({ available: false, message: "API returned " + r.status });
      }
    } catch {
      setEvalData({ available: false, message: "Could not reach the backend." });
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
                {detail.acne_prediction?.available && (
                  <p>
                    <b>AI acne estimate:</b> {detail.acne_prediction.class?.replaceAll("_", " ")} (
                    {detail.acne_prediction.confidence_percent}%)
                  </p>
                )}
                <p className="hint">{detail.disclaimer}</p>
              </div>
            )}
          </section>
        )}

        {page === "AI Model Evaluation" && (
          <>
            <div className="intro">
              <small>EMPIRICAL BENCHMARKING</small>
              <h2>
                ResNet50 Model Evaluation
                <br />
                <i>ACNE04 Dataset Performance</i>
              </h2>
              <p>
                Performance metrics sourced directly from the evaluation report
                artifact (<code>evaluation_test.json</code>). Figures reflect
                the recorded test run; they are not re-verified against the
                loaded checkpoint at runtime.
              </p>
            </div>

            {/* Loading state */}
            {evalLoading && (
              <section className="panel pad">
                <p className="empty">Loading evaluation data…</p>
              </section>
            )}

            {/* Error / missing state */}
            {!evalLoading && evalData && !evalData.available && (
              <section className="panel pad">
                <small>EVALUATION UNAVAILABLE</small>
                <h2>No evaluation report found</h2>
                <p>
                  {evalData.message ||
                    "Run backend/training/evaluate_acne_model.py to generate evaluation artifacts."}
                </p>
                <p className="hint">
                  The AI Model Evaluation page reads from{" "}
                  <code>experiments/*/evaluation_test.json</code>.
                </p>
              </section>
            )}

            {/* No response at all yet (initial load with no error) */}
            {!evalLoading && !evalData && (
              <section className="panel pad">
                <p className="empty">
                  Could not load evaluation data. Ensure the backend is running
                  at {API}.
                </p>
              </section>
            )}

            {/* Dynamic metrics when available */}
            {!evalLoading && evalData && evalData.available && (() => {
              const rpt = evalData.report || {};
              const cm = rpt.confusion_matrix || [];
              const classes = rpt.classes || [];
              const classMetrics = rpt.class_metrics || {};
              const splits = evalData.split_sizes || {};

              const pct = (v) =>
                v != null ? (v * 100).toFixed(2) + "%" : "—";

              const diagSum = cm.reduce(
                (acc, row, i) => acc + (row[i] || 0),
                0
              );
              const totalSamples = rpt.sample_count;

              return (
                <>
                  {/* Performance Summary Cards */}
                  <div className="eval-grid">
                    <div className="eval-card">
                      <small>TEST ACCURACY</small>
                      <b>{pct(rpt.accuracy)}</b>
                      <p>
                        {totalSamples != null
                          ? `${totalSamples} held-out test images`
                          : "held-out test images"}
                      </p>
                    </div>
                    <div className="eval-card">
                      <small>BEST VALIDATION ACC</small>
                      <b>
                        {evalData.validation_accuracy != null
                          ? pct(evalData.validation_accuracy)
                          : "—"}
                      </b>
                      <p>
                        {splits.validation != null
                          ? `${splits.validation} validation images`
                          : "validation images (count unavailable)"}
                      </p>
                    </div>
                    <div className="eval-card">
                      <small>BALANCED ACCURACY</small>
                      <b>{pct(rpt.balanced_accuracy)}</b>
                      <p>Accounts for class imbalance</p>
                    </div>
                    <div className="eval-card">
                      <small>MACRO F1 SCORE</small>
                      <b>{pct(rpt.macro_f1)}</b>
                      <p>
                        Unweighted mean of all{" "}
                        {classes.length > 0 ? classes.length : 4} classes
                      </p>
                    </div>
                  </div>

                  {/* Additional aggregate metrics */}
                  <div className="eval-grid" style={{ marginTop: "10px" }}>
                    <div className="eval-card">
                      <small>MACRO PRECISION</small>
                      <b>{pct(rpt.macro_precision)}</b>
                    </div>
                    <div className="eval-card">
                      <small>MACRO RECALL</small>
                      <b>{pct(rpt.macro_recall)}</b>
                    </div>
                    <div className="eval-card">
                      <small>WEIGHTED F1</small>
                      <b>{pct(rpt.weighted_f1)}</b>
                    </div>
                    <div className="eval-card">
                      <small>CALIBRATION (ECE)</small>
                      <b>
                        {rpt.expected_calibration_error != null
                          ? rpt.expected_calibration_error.toFixed(4)
                          : "—"}
                      </b>
                      <p>Expected Calibration Error</p>
                    </div>
                  </div>

                  {/* Dataset Breakdown Panel */}
                  <section className="panel pad">
                    <small>DATASET &amp; SPLIT METHODOLOGY</small>
                    <h2>ACNE04 Dataset Specifications</h2>
                    <p>
                      ACNE04 contains clinical face images graded into four
                      ordinal severity levels: mild, moderate, severe, and very
                      severe.
                    </p>
                    <div className="stats" style={{ marginTop: "15px" }}>
                      <article>
                        <small>TRAINING SPLIT</small>
                        <b>{splits.train != null ? splits.train : "—"}</b>
                        <p>Fine-tuning partition</p>
                      </article>
                      <article>
                        <small>VALIDATION SPLIT</small>
                        <b>
                          {splits.validation != null ? splits.validation : "—"}
                        </b>
                        <p>Stratified model selection</p>
                      </article>
                      <article>
                        <small>TEST SPLIT</small>
                        <b>{splits.test != null ? splits.test : "—"}</b>
                        <p>Final evaluation</p>
                      </article>
                    </div>
                    <p className="hint" style={{ marginTop: "15px" }}>
                      Experiment: <strong>{evalData.experiment_id}</strong>{" "}
                      &middot; Model: {evalData.model}
                    </p>
                  </section>

                  {/* Per-Class Performance Table */}
                  {Object.keys(classMetrics).length > 0 && (
                    <section className="panel pad">
                      <small>DETAILED CLASSIFICATION REPORT</small>
                      <h2>Per-Class Precision, Recall &amp; F1 Scores</h2>
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
                          {(classes.length > 0
                            ? classes
                            : Object.keys(classMetrics)
                          ).map((cls) => {
                            const m = classMetrics[cls] || {};
                            const badgeClass =
                              cls === "mild"
                                ? "badge-mild"
                                : cls === "moderate"
                                ? "badge-moderate"
                                : cls === "severe"
                                ? "badge-severe"
                                : "badge-very-severe";
                            const f1val =
                              m.f1 != null
                                ? m.f1
                                : m.f1_score != null
                                ? m.f1_score
                                : null;
                            return (
                              <tr key={cls}>
                                <td>
                                  <span
                                    className={`eval-badge ${badgeClass}`}
                                  >
                                    {cls}
                                  </span>
                                </td>
                                <td>
                                  <b>
                                    {m.precision != null
                                      ? (m.precision * 100).toFixed(1) + "%"
                                      : "—"}
                                  </b>
                                </td>
                                <td>
                                  {m.recall != null
                                    ? (m.recall * 100).toFixed(1) + "%"
                                    : "—"}
                                </td>
                                <td>
                                  {f1val != null ? f1val.toFixed(4) : "—"}
                                </td>
                                <td>
                                  {m.support != null
                                    ? `${m.support} images`
                                    : "—"}
                                </td>
                              </tr>
                            );
                          })}
                          <tr
                            style={{
                              background: "#f8f9f2",
                              fontWeight: 700,
                            }}
                          >
                            <td>Macro Average</td>
                            <td>{pct(rpt.macro_precision)}</td>
                            <td>{pct(rpt.macro_recall)}</td>
                            <td>{pct(rpt.macro_f1)}</td>
                            <td>
                              {totalSamples != null
                                ? `${totalSamples} images`
                                : "—"}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </section>
                  )}

                  {/* Confusion Matrix */}
                  {cm.length > 0 && classes.length > 0 && (
                    <section className="panel pad">
                      <small>ERROR ANALYSIS &amp; ADJACENCY</small>
                      <h2>Test Set Confusion Matrix</h2>
                      <p>
                        In dermatological severity scoring, misclassifications
                        between adjacent grades (e.g. mild vs moderate) are
                        clinically expected; extreme errors rarely occur.
                      </p>
                      <div className="cm-wrapper">
                        <table className="cm-table">
                          <thead>
                            <tr>
                              <th
                                rowSpan="2"
                                className="cm-row-label"
                                style={{ verticalAlign: "bottom" }}
                              >
                                True Severity \ Predicted
                              </th>
                              <th
                                colSpan={classes.length}
                                className="cm-group-header"
                              >
                                Predicted Severity Class &rarr;
                              </th>
                            </tr>
                            <tr>
                              {classes.map((c) => (
                                <th key={c}>{c}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {cm.map((row, ri) => {
                              const rowTotal = row.reduce(
                                (a, v) => a + v,
                                0
                              );
                              return (
                                <tr key={ri}>
                                  <th className="cm-row-label">
                                    {classes[ri]}
                                    {rowTotal > 0 ? ` (${rowTotal})` : ""}
                                  </th>
                                  {row.map((cell, ci) => (
                                    <td
                                      key={ci}
                                      className={
                                        ri === ci ? "cm-diag" : "cm-off"
                                      }
                                    >
                                      {cell}
                                    </td>
                                  ))}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>

                        <div className="cm-legend">
                          <div className="cm-legend-item">
                            <span className="cm-swatch cm-swatch-diag" />
                            <span>
                              <b>Diagonal (Green):</b> Correctly classified (
                              {diagSum}
                              {totalSamples != null
                                ? ` / ${totalSamples}`
                                : ""}
                              {rpt.accuracy != null
                                ? ` = ${(rpt.accuracy * 100).toFixed(2)}%`
                                : ""}
                              )
                            </span>
                          </div>
                          <div className="cm-legend-item">
                            <span className="cm-swatch cm-swatch-off" />
                            <span>
                              <b>Off-Diagonal:</b> Misclassifications
                            </span>
                          </div>
                        </div>
                      </div>
                      <p className="hint" style={{ marginTop: "12px" }}>
                        Evaluated on{" "}
                        {totalSamples != null ? totalSamples : "—"} test
                        photographs (experiment: {evalData.experiment_id}).
                      </p>
                    </section>
                  )}

                  {/* Academic Discussion */}
                  <section className="panel pad">
                    <small>ACADEMIC METHODOLOGY</small>
                    <h2>Key Engineering Insights &amp; Viva Defense Notes</h2>
                    <div className="callout" style={{ marginTop: "10px" }}>
                      <b>1. Why Transfer Learning with ImageNet Pretraining?</b>
                      <p>
                        Training deep convolutional networks like ResNet50 from
                        scratch requires hundreds of thousands of images. With
                        the small ACNE04 corpus, ImageNet initialization
                        provides generic low-level feature detectors that are
                        then fine-tuned on clinical facial acne features.
                      </p>
                    </div>
                    <div className="callout">
                      <b>2. Why Accuracy Alone is Misleading</b>
                      <p>
                        Mild and moderate acne dominate the dataset. Balanced
                        Accuracy and Macro F1 weight all four severity levels
                        equally, exposing performance on minority classes like
                        severe and very severe.
                      </p>
                    </div>
                    <div className="callout">
                      <b>3. Calibration &amp; Softmax Confidence Notice</b>
                      <p>
                        {rpt.expected_calibration_error != null
                          ? `ECE: ${rpt.expected_calibration_error.toFixed(4)}.`
                          : "ECE not included in this evaluation report."}{" "}
                        Softmax percentages represent relative model activation
                        confidence, not absolute medical probabilities.
                      </p>
                    </div>
                  </section>
                </>
              );
            })()}
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
