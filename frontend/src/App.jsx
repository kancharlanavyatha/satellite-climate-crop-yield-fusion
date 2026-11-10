import { useEffect, useRef, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  BarChart, Bar, Cell, PieChart, Pie,
} from "recharts";

const API = "http://localhost:8000/api";
const SLIDERS = {
  HeatStressDays_gt35C: "Heat stress days (>35°C)",
  Rain_Total_mm: "Total rainfall (mm)",
  Temp_Mean_C: "Mean temperature (°C)",
  Soil_pH: "Soil pH",
  Clay_pct: "Clay (%)",
};
const MODALITY = [
  { name: "Weather & Climate", value: 55.03, color: "#38bdf8" },
  { name: "Satellite Indices", value: 23.19, color: "#4ade80" },
  { name: "Soil & Terrain", value: 20.47, color: "#fbbf24" },
  { name: "Sown Area", value: 1.31, color: "#a78bfa" },
];
const BENCH = [
  { model: "YieldNet", MAE: 0.3951, R2: 0.4496 },
  { model: "XGBoost", MAE: 0.451, R2: 0.3203 },
  { model: "LightGBM", MAE: 0.4431, R2: 0.3027 },
  { model: "RandomForest", MAE: 0.4564, R2: 0.2867 },
  { model: "Ridge", MAE: 0.5037, R2: 0.1315 },
];
const tip = { background: "#0f1f1a", border: "1px solid #24413a", borderRadius: 8, color: "#e8f5ee" };

const Card = ({ title, children, className = "" }) => (
  <section className={`card ${className}`}>
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

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">🌾 <span>YieldSight</span></div>
        <p className="muted">Satellite + climate crop yield forecasting</p>

        <label>District</label>
        <select value={district} onChange={(e) => pickDistrict(e.target.value)}>
          {meta.districts.map((d) => <option key={d}>{d}</option>)}
        </select>

        <label>Year</label>
        <div className="chips">
          {(meta.years[district] || []).map((y) => (
            <button key={y} className={y === year ? "chip on" : "chip"}
              onClick={() => { setYear(y); setOverrides({}); }}>{y}</button>
          ))}
        </div>

        <div className="what-if">
          <div className="row"><label>What-if scenario</label>
            <button className="link" onClick={() => setOverrides({})}>Reset</button></div>
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
        {error && <div className="error">{error}</div>}
        {data && (
          <>
            <div className="kpis">
              <div className={`kpi hero ${loading ? "pulse" : ""}`}>
                <span>Predicted yield</span>
                <strong>{data.prediction.toFixed(2)}<small> t/ha</small></strong>
                <em>Likely range {(data.prediction - data.mae).toFixed(2)} – {(data.prediction + data.mae).toFixed(2)}</em>
              </div>
              <div className="kpi"><span>Actual yield</span>
                <strong>{data.actual.toFixed(2)}<small> t/ha</small></strong>
                <em>{district}, {year}</em></div>
              <div className="kpi"><span>Prediction error</span>
                <strong className={Math.abs(err) <= data.mae ? "good" : "warn"}>
                  {err >= 0 ? "+" : ""}{err.toFixed(2)}<small> t/ha</small></strong>
                <em>Model MAE {data.mae}</em></div>
            </div>

            <div className="grid">
              <Card title="Satellite view (Sentinel-2)" className="span2">
                <div className="imgs">
                  {[["rgb", "True colour"], ["ndvi", "NDVI · vegetation"], ["ndwi", "NDWI · water"]].map(([k, l]) => (
                    <figure key={k}><img src={`data:image/png;base64,${data.images[k]}`} alt={l} /><figcaption>{l}</figcaption></figure>
                  ))}
                </div>
              </Card>

              <Card title={`${district}: actual vs predicted`}>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={data.history}>
                    <CartesianGrid stroke="#1d3a32" strokeDasharray="3 3" />
                    <XAxis dataKey="year" stroke="#7fa596" />
                    <YAxis stroke="#7fa596" domain={["auto", "auto"]} />
                    <Tooltip contentStyle={tip} /><Legend />
                    <Line dataKey="actual" stroke="#4ade80" strokeWidth={2.5} dot />
                    <Line dataKey="predicted" stroke="#38bdf8" strokeWidth={2.5} strokeDasharray="5 4" dot />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card title={`All districts · ${year} (actual)`}>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data.districts} layout="vertical" margin={{ left: 30 }}>
                    <XAxis type="number" stroke="#7fa596" />
                    <YAxis type="category" dataKey="district" stroke="#7fa596" width={90} fontSize={11} />
                    <Tooltip contentStyle={tip} />
                    <Bar dataKey="actual" radius={4}>
                      {data.districts.map((d) => <Cell key={d.district} fill={d.district === district ? "#fbbf24" : "#2f7d62"} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="What drives the model">
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={MODALITY} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} paddingAngle={2}>
                      {MODALITY.map((m) => <Cell key={m.name} fill={m.color} stroke="none" />)}
                    </Pie>
                    <Tooltip contentStyle={tip} formatter={(v) => `${v}%`} /><Legend />
                  </PieChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Benchmark (MAE, lower is better)">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={BENCH}>
                    <CartesianGrid stroke="#1d3a32" strokeDasharray="3 3" />
                    <XAxis dataKey="model" stroke="#7fa596" fontSize={11} />
                    <YAxis stroke="#7fa596" domain={[0.3, 0.55]} />
                    <Tooltip contentStyle={tip} />
                    <Bar dataKey="MAE" radius={4}>
                      {BENCH.map((b) => <Cell key={b.model} fill={b.model === "YieldNet" ? "#4ade80" : "#3b5f54"} />)}
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