import { useState } from 'react'
import { motion } from 'framer-motion'
import { Database, Trophy, Zap, HardDrive } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
} from 'recharts'
import { creditData, CREDIT_MODEL_COLORS } from '../data/credit'
import { CATEGORY_COLORS, CATEGORY_LABELS } from '../data/churn'

// Sort models by AUC descending
const sortedModels = [...creditData.models].sort(
  (a, b) => b.metrics.auc_roc - a.metrics.auc_roc
)

// All unique sample sizes across models
const allSampleSizes = [
  ...new Set(creditData.models.flatMap((m) => m.scaling.map((s) => s.n_samples))),
].sort((a, b) => a - b)

// Prepare scaling data — models with missing points get undefined (gap in line)
const scalingData = allSampleSizes.map((n) => {
  const point: Record<string, number | undefined> = { n_samples: n }
  creditData.models.forEach((m) => {
    const match = m.scaling.find((sc) => sc.n_samples === n)
    point[m.name] = match?.auc
  })
  return point
})

// AUC bar chart data
const aucBarData = sortedModels.map((m) => ({
  name: m.name.replace(' (Default)', ' ⓓ').replace(' (Tuned)', ' ⓣ'),
  fullName: m.name,
  auc: m.metrics.auc_roc,
  category: m.category,
}))

// Training time bar chart data
const timeBarData = [...sortedModels]
  .sort((a, b) => a.metrics.train_time_sec - b.metrics.train_time_sec)
  .map((m) => ({
    name: m.name.replace(' (Default)', ' ⓓ').replace(' (Tuned)', ' ⓣ'),
    fullName: m.name,
    time: m.metrics.train_time_sec,
    category: m.category,
  }))

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-zinc-900/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
        <p className="text-zinc-400 text-xs font-mono mb-2">
          {label.toLocaleString()} samples
        </p>
        {payload
          .sort((a: any, b: any) => b.value - a.value)
          .map((entry: any) => (
            <div
              key={entry.name}
              className="flex items-center gap-2 text-sm py-0.5"
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: entry.color }}
              />
              <span className="text-zinc-300">{entry.name}</span>
              <span className="text-zinc-100 font-mono font-medium ml-auto pl-4">
                {entry.value.toFixed(4)}
              </span>
            </div>
          ))}
      </div>
    )
  }
  return null
}

const BarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-zinc-900/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
        <p className="text-zinc-200 text-sm font-medium">{data.fullName}</p>
        <p className="text-zinc-100 font-mono text-lg font-bold">
          {payload[0].value.toFixed(4)}
        </p>
      </div>
    )
  }
  return null
}

const TimeBarTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <div className="bg-zinc-900/95 backdrop-blur-xl border border-white/10 rounded-xl px-4 py-3 shadow-2xl">
        <p className="text-zinc-200 text-sm font-medium">{data.fullName}</p>
        <p className="text-zinc-100 font-mono text-lg font-bold">
          {payload[0].value}s
        </p>
      </div>
    )
  }
  return null
}

export function CreditRiskTab() {
  const [hoveredModel, setHoveredModel] = useState<string | null>(null)
  const best = sortedModels[0]
  const fastest = [...creditData.models].sort(
    (a, b) => a.metrics.train_time_sec - b.metrics.train_time_sec
  )[0]
  const lightest = [...creditData.models].sort(
    (a, b) => a.metrics.peak_memory_mb - b.metrics.peak_memory_mb
  )[0]

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      {/* Dataset Header */}
      <div className="flex items-start gap-4">
        <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
          <Database className="w-6 h-6 text-zinc-400" />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-zinc-100 tracking-tight">
            Home Credit Default Risk
          </h2>
          <p className="text-zinc-500 mt-1 font-mono text-sm">
            {creditData.n_samples.toLocaleString()} rows ·{' '}
            {creditData.n_features} features ·{' '}
            {(creditData.target_rate * 100).toFixed(1)}% positive · Kaggle
            Competition
          </p>
        </div>
      </div>

      {/* Winner Highlights */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl p-5 border-l-4 border-l-amber-500">
          <div className="flex items-center gap-2 mb-2">
            <Trophy className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
              Best AUC-ROC
            </span>
          </div>
          <span className="text-2xl font-bold font-mono text-zinc-100">
            {best.metrics.auc_roc}
          </span>
          <span className="text-sm text-amber-400 ml-2">{best.name}</span>
        </div>

        <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl p-5 border-l-4 border-l-emerald-500">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
              Fastest Training
            </span>
          </div>
          <span className="text-2xl font-bold font-mono text-zinc-100">
            {fastest.metrics.train_time_sec}s
          </span>
          <span className="text-sm text-emerald-400 ml-2">{fastest.name}</span>
        </div>

        <div className="bg-white/[0.02] border border-white/[0.05] rounded-xl p-5 border-l-4 border-l-sky-500">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-4 h-4 text-sky-400" />
            <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
              Lightest Memory
            </span>
          </div>
          <span className="text-2xl font-bold font-mono text-zinc-100">
            {lightest.metrics.peak_memory_mb} MB
          </span>
          <span className="text-sm text-sky-400 ml-2">{lightest.name}</span>
        </div>
      </motion.div>

      {/* Leaderboard Table */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-2xl overflow-hidden"
      >
        <div className="px-6 py-4 border-b border-white/[0.05]">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
            Leaderboard
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-zinc-500 text-xs uppercase tracking-wider">
                <th className="text-left px-6 py-3 font-medium">#</th>
                <th className="text-left px-6 py-3 font-medium">Model</th>
                <th className="text-left px-6 py-3 font-medium">Type</th>
                <th className="text-right px-6 py-3 font-medium">AUC-ROC</th>
                <th className="text-right px-6 py-3 font-medium">Log Loss</th>
                <th className="text-right px-6 py-3 font-medium">Train (s)</th>
                <th className="text-right px-6 py-3 font-medium">
                  Inf (ms/1k)
                </th>
                <th className="text-right px-6 py-3 font-medium">Mem (MB)</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((model, i) => (
                <tr
                  key={model.name}
                  className={`border-t border-white/[0.03] transition-colors ${
                    hoveredModel === model.name
                      ? 'bg-white/[0.04]'
                      : 'hover:bg-white/[0.02]'
                  }`}
                  onMouseEnter={() => setHoveredModel(model.name)}
                  onMouseLeave={() => setHoveredModel(null)}
                >
                  <td className="px-6 py-3.5">
                    <span
                      className={`font-mono font-bold ${
                        i === 0
                          ? 'text-amber-400'
                          : i === 1
                            ? 'text-zinc-400'
                            : i === 2
                              ? 'text-orange-600'
                              : 'text-zinc-600'
                      }`}
                    >
                      {i + 1}
                    </span>
                  </td>
                  <td className="px-6 py-3.5">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{
                          background: CREDIT_MODEL_COLORS[model.name],
                        }}
                      />
                      <span className="text-zinc-200 font-medium">
                        {model.name}
                      </span>
                      {model.tuned && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-violet-500/10 text-violet-400 border border-violet-500/20">
                          tuned
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-3.5">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full border ${CATEGORY_COLORS[model.category]}`}
                    >
                      {CATEGORY_LABELS[model.category]}
                    </span>
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono">
                    <span
                      className={
                        i === 0
                          ? 'text-amber-400 font-bold'
                          : 'text-zinc-300'
                      }
                    >
                      {model.metrics.auc_roc.toFixed(4)}
                    </span>
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono text-zinc-400">
                    {model.metrics.log_loss.toFixed(4)}
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono text-zinc-400">
                    {model.metrics.train_time_sec}
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono text-zinc-400">
                    {model.metrics.inference_time_ms_per_1k > 1000
                      ? `${(model.metrics.inference_time_ms_per_1k / 1000).toFixed(1)}s`
                      : model.metrics.inference_time_ms_per_1k}
                  </td>
                  <td className="px-6 py-3.5 text-right font-mono text-zinc-400">
                    {model.metrics.peak_memory_mb}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AUC Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-2xl p-6"
        >
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-6">
            AUC-ROC Comparison
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={aucBarData}
              layout="vertical"
              margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.04)"
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0.6, 0.8]}
                tick={{ fill: '#71717a', fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(2)}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#a1a1aa', fontSize: 11 }}
                width={130}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<BarTooltip />}
                cursor={{ fill: 'rgba(255,255,255,0.02)' }}
              />
              <Bar dataKey="auc" radius={[0, 6, 6, 0]} barSize={24}>
                {aucBarData.map((entry) => (
                  <Cell
                    key={entry.fullName}
                    fill={CREDIT_MODEL_COLORS[entry.fullName]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Training Time Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-2xl p-6"
        >
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-6">
            Training Time (seconds)
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={timeBarData}
              layout="vertical"
              margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(255,255,255,0.04)"
                horizontal={false}
              />
              <XAxis
                type="number"
                tick={{ fill: '#71717a', fontSize: 11 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#a1a1aa', fontSize: 11 }}
                width={130}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<TimeBarTooltip />}
                cursor={{ fill: 'rgba(255,255,255,0.02)' }}
              />
              <Bar dataKey="time" radius={[0, 6, 6, 0]} barSize={24}>
                {timeBarData.map((entry) => (
                  <Cell
                    key={entry.fullName}
                    fill={CREDIT_MODEL_COLORS[entry.fullName]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Scaling Curves */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-2xl p-6"
      >
        <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-6">
          Scaling Curves — AUC vs Training Samples
        </h3>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={scalingData}
            margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.04)"
            />
            <XAxis
              dataKey="n_samples"
              tick={{ fill: '#71717a', fontSize: 11 }}
              tickFormatter={(v: number) => v >= 1000 ? `${v / 1000}k` : String(v)}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <YAxis
              domain={[0.48, 0.8]}
              tick={{ fill: '#71717a', fontSize: 11 }}
              tickFormatter={(v: number) => v.toFixed(2)}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              height={40}
              iconType="circle"
              iconSize={8}
              formatter={(value: string) => (
                <span className="text-zinc-400 text-xs">{value}</span>
              )}
            />
            {creditData.models.map((model) => (
              <Line
                key={model.name}
                type="monotone"
                dataKey={model.name}
                stroke={CREDIT_MODEL_COLORS[model.name]}
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 0, fill: CREDIT_MODEL_COLORS[model.name] }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: '#0a0f1a' }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Key Takeaways */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
        className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-2xl p-6 space-y-4"
      >
        <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">
          Key Takeaways
        </h3>
        <div className="space-y-3">
          <div className="flex gap-3 items-start">
            <span className="text-amber-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">AutoGluon edges out the field</strong>{' '}
              with 0.7857 AUC-ROC, but at enormous cost — 650s training and 2.6 GB RAM.
              XGBoost (Tuned) matches at 0.7844 in 180s, making it the better
              production choice.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-indigo-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">
                Tuning is critical on high-dimensional data
              </strong>{' '}
              — XGBoost jumps from 0.7551 → 0.7844 (+3.9%) with Optuna.
              On 302 features with heavy missingness, default hyperparameters
              leave significant performance on the table.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-emerald-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">
                CatBoost shines out of the box
              </strong>{' '}
              — 0.7798 AUC with zero tuning, native categorical handling, and
              automatic class weighting. Best default-config model by a wide margin.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-rose-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">
                TabPFN hits its ceiling at scale
              </strong>{' '}
              — scaling stops at 50k samples (0.7733) and can't leverage the full
              307k dataset. Its 6s/1k inference and 1.2 GB memory make it
              impractical for production credit scoring.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-sky-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">
                FT-Transformer collapses — but it's a preprocessing issue
              </strong>{' '}
              — 0.6289 AUC with simple median imputation on 302 sparse features.
              The per-feature linear embeddings can't handle heavy missingness
              or raw categoricals. A fairer test would need proper NaN masking
              and learned categorical embeddings.
            </p>
          </div>
          <div className="flex gap-3 items-start">
            <span className="text-violet-400 mt-0.5">→</span>
            <p className="text-zinc-300 text-sm leading-relaxed">
              <strong className="text-zinc-100">
                Data volume matters more than model choice
              </strong>{' '}
              — every GBDT model gains 6–8% AUC going from 1k to 245k samples.
              The scaling curves show no signs of saturation, suggesting even
              more data would help.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}
