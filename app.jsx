import { useEffect, useRef, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  BarChart, Bar, Cell, PieChart, Pie,
} from "recharts";

const API =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";
const SLIDERS = {
  HeatStressDays_gt35C: "Heat stress days (>35°C)",
  Rain_Total_mm: "Total rainfall (mm)",
  Temp_Mean_C: "Mean temperature (°C)",
  Soil_pH: "Soil pH",
  Clay_pct: "Clay (%)",
};

const MODALITY = [
  { name: "Weather & Climate", value: 55.03, color: "#14532d" },
  { name: "Satellite Indices", value: 23.19, color: "#2f855a" },
  { name: "Soil & Terrain", value: 20.47, color: "#68d391" },
  { name: "Sown Area", value: 1.31, color: "#c6f6d5" },
];
const BENCH = [
  { model: "YieldNet", MAE: 0.3951 },
  { model: "XGBoost", MAE: 0.451 },
  { model: "LightGBM", MAE: 0.4431 },
  { model: "RandForest", MAE: 0.4564 },
  { model: "Ridge", MAE: 0.5037 },
];
const AX = "#5b7a68";
const GRID = "#dbe8dc";
const tip = {
  background: "#fff", border: "1px solid #cfe3d2", borderRadius: 12, color: "#10281c",
  boxShadow: "0 10px 30px -10px rgba(16,40,28,.25)",
};

function useCountUp(target, ms = 800) {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now(), a = from.current;
    let raf;
    const tick = (t) => {
      const p = Math.min((t - start) / ms, 1);
      const e = 1 - Math.pow(1 - p, 3);
      setV(a + (target - a) * e);
      if (p < 1) raf = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

const Card = ({ title, children, className = "", delay = 0 }) => (
  <section className={`card ${className}`} style={{ animationDelay: `${delay}ms` }}>
    {title && <h3>{title}</h3>}
    {children}
  </section>
);

export default function App() {
  const [meta, setMeta] = useState(null);
  const [district, setDistrict] = useState("");
  const [year, setYear] = useState(null);
  const [overrides, setOverrides] = useState({});
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const timer = useRef();
  const shown = useCountUp(data?.prediction ?? 0);

  useEffect(() => {
    fetch(`${API}/meta`)
      .then((r) => r.json())
      .then((m) => {
        setMeta(m);
        const d = m.districts[0];
        setDistrict(d);
        setYear(m.years[d][0]);
      })
      .catch(() => setError("Cannot reach API. Is `uvicorn api:app --port 8000` running?"));
  }, []);

  useEffect(() => {
    if (!district || !year) return;
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await fetch(`${API}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ district, year, overrides }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        setData(await r.json());
        setError("");
      } catch (e) {
        setError(String(e.message || e));
      }
      setLoading(false);
    }, 250);
  }, [district, year, overrides]);

  const pickDistrict = (d) => {
    setDistrict(d);
    setYear(meta.years[d][0]);
    setOverrides({});
  };

  if (!meta) return <div className="boot">{error || "Loading…"}</div>;

  const err = data ? data.prediction - data.actual : 0;
  const changed = Object.keys(overrides).length > 0;

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">Yield<span>Sight</span></div>
        <p className="tag">Satellite &amp; climate intelligence for crop yield</p>

        <label>District</label>
        <select value={district} onChange={(e) => pickDistrict(e.target.value)}>
          {meta.districts.map((d) => <option key={d}>{d}</option>)}
        </select>

        <label>Season</label>
        <div className="chips">
          {(meta.years[district] || []).map((y) => (
            <button key={y} className={y === year ? "chip on" : "chip"}
              onClick={() => { setYear(y); setOverrides({}); }}>{y}</button>
          ))}
        </div>

        <div className="what-if">
          <div className="row">
            <label style={{ margin: 0 }}>What-if scenario</label>
            <button className="link" onClick={() => setOverrides({})} disabled={!changed}>Reset</button>
          </div>
          {Object.entries(SLIDERS).map(([k, label]) => {
            const r = meta.ranges[k];
            const v = overrides[k] ?? data?.base?.[k] ?? r.min;
            return (
              <div key={k} className="slider">
                <div className="row"><span>{label}</span><b>{Number(v).toFixed(1)}</b></div>
                <input type="range" min={r.min} max={r.max} step={(r.max - r.min) / 200}
                  value={v} onChange={(e) => setOverrides({ ...overrides, [k]: +e.target.value })} />
              </div>
            );
          })}
        </div>
      </aside>

      <main className="main">
        <header className="head">
          <div>
            <p className="eyebrow">Yield forecast</p>
            <h1>{district}<em> · {year}</em></h1>
          </div>
          <span className={`badge ${loading ? "busy" : ""}`}>
            <i />{loading ? "Computing" : changed ? "Scenario" : "Live model"}
          </span>
        </header>

        {error && <div className="error">{error}</div>}
        {data && (
          <>
            <div className="kpis">
              <div className="kpi hero" style={{ animationDelay: "0ms" }}>
                <span>Predicted yield</span>
                <strong>{shown.toFixed(2)}<small> t/ha</small></strong>
                <em>Likely range {(data.prediction - data.mae).toFixed(2)} – {(data.prediction + data.mae).toFixed(2)}</em>
              </div>
              <div className="kpi" style={{ animationDelay: "80ms" }}>
                <span>Actual yield</span>
                <strong>{data.actual.toFixed(2)}<small> t/ha</small></strong>
                <em>Recorded for {year}</em>
              </div>
              <div className="kpi" style={{ animationDelay: "160ms" }}>
                <span>Prediction error</span>
                <strong className={Math.abs(err) <= data.mae ? "good" : "warn"}>
                  {err >= 0 ? "+" : ""}{err.toFixed(2)}<small> t/ha</small></strong>
                <em>Model MAE {data.mae}</em>
              </div>
            </div>

            <div className="grid">
              <Card title="Satellite view · Sentinel-2" className="span2 dark" delay={200}>
                <div className="imgs">
                  {[["rgb", "True colour"], ["ndvi", "NDVI · vegetation"], ["ndwi", "NDWI · water"]].map(([k, l]) => (
                    <figure key={k}>
                      <img src={`data:image/png;base64,${data.images[k]}`} alt={l} />
                      <figcaption>{l}</figcaption>
                    </figure>
                  ))}
                </div>
              </Card>

              <Card title={`${district} · actual vs predicted`} delay={280}>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={data.history}>
                    <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                    <XAxis dataKey="year" stroke={AX} />
                    <YAxis stroke={AX} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={tip} /><Legend />
                    <Line dataKey="actual" stroke="#14532d" strokeWidth={3} dot={{ r: 4 }} />
                    <Line dataKey="predicted" stroke="#48bb78" strokeWidth={3} strokeDasharray="6 4" dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card title={`All districts · ${year}`} delay={340}>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data.districts} layout="vertical" margin={{ left: 30 }}>
                    <XAxis type="number" stroke={AX} />
                    <YAxis type="category" dataKey="district" stroke={AX} width={90} fontSize={11} />
                    <Tooltip contentStyle={tip} cursor={{ fill: "#eef5ee" }} />
                    <Bar dataKey="actual" radius={6}>
                      {data.districts.map((d) => (
                        <Cell key={d.district} fill={d.district === district ? "#14532d" : "#a7d3b4"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="What drives the model" delay={400}>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={MODALITY} dataKey="value" nameKey="name" innerRadius={62} outerRadius={95} paddingAngle={3}>
                      {MODALITY.map((m) => <Cell key={m.name} fill={m.color} stroke="none" />)}
                    </Pie>
                    <Tooltip contentStyle={tip} formatter={(v) => `${v}%`} /><Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Benchmark · MAE (lower is better)" delay={460}>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={BENCH}>
                    <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="model" stroke={AX} fontSize={11} />
                    <YAxis stroke={AX} domain={[0.3, 0.55]} />
                    <Tooltip contentStyle={tip} cursor={{ fill: "#eef5ee" }} />
                    <Bar dataKey="MAE" radius={[8, 8, 0, 0]}>
                      {BENCH.map((b) => <Cell key={b.model} fill={b.model === "YieldNet" ? "#14532d" : "#b7d9c2"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          </>
        )}
      </main>
    </div>
  );
}