import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Brain, Clock, Database, Route, Send, Sparkles } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const EMPTY_RESULT = {
  path: [],
  fanout: {},
  checkpoint: {},
  summarization: {},
  agent: {},
  metrics: { tokens: {} },
};

function JsonBlock({ value }) {
  if (value === undefined || value === null) {
    return <div className="empty">Not available</div>;
  }
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}

function Panel({ title, icon, children }) {
  return (
    <section className="panel">
      <div className="panel-title">
        {icon}
        <span>{title}</span>
      </div>
      {children}
    </section>
  );
}

function PathStrip({ path }) {
  return (
    <div className="path-strip">
      <div className="path-label">
        <Route size={18} />
        <span>Path Taken</span>
      </div>
      <div className="path-nodes">
        {(path || []).map((node, index) => (
          <React.Fragment key={`${node}-${index}`}>
            <span className="path-node">{node}</span>
            {index < path.length - 1 && <span className="path-arrow">/</span>}
          </React.Fragment>
        ))}
        {!path?.length && <span className="path-empty">Run a question to see execution path</span>}
      </div>
    </div>
  );
}

function VerdictCard({ verdict, route, memoryAnswer }) {
  if (route === "memory") {
    return (
      <section className="verdict-card">
        <div className="verdict-kicker">Memory Answer</div>
        <div className="decision decision-memory">Memory</div>
        <div className="confidence">Thread context</div>
        <p>{memoryAnswer || "No prior product research was found for this thread."}</p>
      </section>
    );
  }

  const decision = verdict?.decision || "Waiting";
  return (
    <section className="verdict-card">
      <div className="verdict-kicker">Final Verdict</div>
      <div className={`decision decision-${decision.toLowerCase().replace(/[^a-z]/g, "")}`}>
        {decision}
      </div>
      <div className="confidence">{verdict?.confidence || "No run yet"}</div>
      <p>{verdict?.reasoning || "Ask a launch research question to generate a founder-facing verdict."}</p>
      <div className="factors">
        {(verdict?.key_factors || []).map((factor, index) => (
          <span key={index}>{factor}</span>
        ))}
      </div>
    </section>
  );
}

function MetricBar({ result }) {
  const tokens = result.metrics?.tokens || {};
  return (
    <div className="metrics">
      <div>
        <Clock size={16} />
        <span>{result.metrics?.latency_ms ?? "-"} ms</span>
      </div>
      <div>
        <Activity size={16} />
        <span>Total tokens: {tokens.total ?? "n/a"}</span>
      </div>
      <div>
        <Brain size={16} />
        <span>Prompt: {tokens.prompt ?? "n/a"} / Completion: {tokens.completion ?? "n/a"}</span>
      </div>
    </div>
  );
}

function App() {
  const [threadId, setThreadId] = React.useState("default-thread");
  const [question, setQuestion] = React.useState("Research launching vegan cosmetics in UAE");
  const [result, setResult] = React.useState(EMPTY_RESULT);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  async function runGraph(event) {
    event.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId, question }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Request failed");
      }
      setResult(payload);
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <h1>LaunchLens</h1>
          <p>Market research graph console</p>
        </div>
        <MetricBar result={result} />
      </header>

      <PathStrip path={result.path} />

      <form className="query-bar" onSubmit={runGraph}>
        <input
          value={threadId}
          onChange={(event) => setThreadId(event.target.value)}
          aria-label="Thread ID"
        />
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          aria-label="Question"
        />
        <button type="submit" disabled={loading}>
          <Send size={18} />
          <span>{loading ? "Running" : "Run"}</span>
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      <section className="main-grid">
        <div className="left-column">
          <Panel title="Checkpointer" icon={<Database size={18} />}>
            <dl>
              <dt>Thread</dt>
              <dd>{result.checkpoint?.thread_id || threadId}</dd>
              <dt>Database</dt>
              <dd>{result.checkpoint?.db_path || "-"}</dd>
              <dt>Messages</dt>
              <dd>{result.checkpoint?.message_count ?? "-"}</dd>
            </dl>
          </Panel>

          <Panel title="Summarization" icon={<Brain size={18} />}>
            <div className="summary-text">
              {result.summarization?.summary || "No running summary yet."}
            </div>
            <div className="subtle">{result.summarization?.summary_chars || 0} chars</div>
          </Panel>
        </div>

        <div className="center-column">
          <VerdictCard
            verdict={result.verdict}
            route={result.route}
            memoryAnswer={result.agent?.answer || result.answer}
          />
          <Panel title="Agent Node" icon={<Sparkles size={18} />}>
            <p className="agent-answer">{result.agent?.answer || "No agent output yet."}</p>
            <div className="subtle">Tool calls: {result.agent?.tool_calls ?? 0}</div>
          </Panel>
        </div>

        <div className="right-column">
          <Panel title="Router" icon={<Route size={18} />}>
            <dl>
              <dt>Route</dt>
              <dd>{result.route || "-"}</dd>
              <dt>Search query</dt>
              <dd>{result.search_query || "-"}</dd>
              <dt>Region</dt>
              <dd>{result.target_region || "-"}</dd>
            </dl>
            <div className="subtle">{result.route_reason || ""}</div>
          </Panel>

          <Panel title="Fan-Out Results" icon={<Activity size={18} />}>
            <div className="tabs-grid">
              <details open>
                <summary>Google Trends</summary>
                <JsonBlock value={result.fanout?.trends} />
              </details>
              <details open>
                <summary>Amazon Search</summary>
                <JsonBlock value={result.fanout?.amazon_search} />
              </details>
              <details>
                <summary>Amazon Products</summary>
                <JsonBlock value={result.fanout?.amazon_products} />
              </details>
              <details>
                <summary>Google News</summary>
                <JsonBlock value={result.fanout?.news} />
              </details>
            </div>
          </Panel>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
