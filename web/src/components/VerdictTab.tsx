import { motion } from 'framer-motion'
import { Gavel, Sparkles, Timer, TrendingUp } from 'lucide-react'
import { churnData } from '../data/churn'
import { creditData } from '../data/credit'
import { fraudData } from '../data/fraud'

type DatasetPack = {
  label: string
  data: typeof churnData
}

const datasets: DatasetPack[] = [
  { label: 'Churn', data: churnData },
  { label: 'Credit', data: creditData },
  { label: 'Fraud', data: fraudData },
]

function bestModel(models: typeof churnData.models) {
  return [...models].sort((a, b) => b.metrics.auc_roc - a.metrics.auc_roc)[0]
}

function categoryAverage(models: typeof churnData.models, category: string) {
  const arr = models.filter((m) => m.category === category)
  if (!arr.length) return 0
  return arr.reduce((s, m) => s + m.metrics.auc_roc, 0) / arr.length
}

export function VerdictTab() {
  const winners = datasets.map((d) => ({
    dataset: d.label,
    winner: bestModel(d.data.models),
  }))

  const allModels = [...churnData.models, ...creditData.models, ...fraudData.models]
  const fastest = [...allModels].sort(
    (a, b) => a.metrics.train_time_sec - b.metrics.train_time_sec
  )[0]
  const bestOverall = [...allModels].sort((a, b) => b.metrics.auc_roc - a.metrics.auc_roc)[0]

  const catRows = [
    {
      name: 'Gradient Boosting',
      key: 'gradient_boosting',
      verdict: 'Most robust across all scales',
    },
    {
      name: 'AutoML',
      key: 'automl',
      verdict: 'Strong AUC, expensive runtime/memory',
    },
    {
      name: 'Foundation Model',
      key: 'foundation_model',
      verdict: 'Best in low-data settings, weaker at scale',
    },
    {
      name: 'Deep Learning',
      key: 'deep_learning',
      verdict: 'Potentially good, less consistent here',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      <div className="glass-panel rounded-2xl p-6 md:p-7">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl border border-white/10 bg-white/5">
            <Gavel className="w-5 h-5 text-violet-300" />
          </div>
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-zinc-100 tracking-tight">
              Final Verdict
            </h2>
            <p className="text-zinc-300 mt-2 leading-relaxed max-w-3xl">
              Foundation-style models can read tabular data and stay competitive, especially on smaller
              datasets. But on large, imbalanced, production-like regimes, tuned gradient boosting still
              provides the most reliable accuracy-performance tradeoff.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" /> Highest AUC observed
          </div>
          <p className="text-2xl font-mono text-zinc-100 font-bold">{bestOverall.metrics.auc_roc.toFixed(4)}</p>
          <p className="text-sm text-zinc-400 mt-1">{bestOverall.name}</p>
        </div>
        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mb-2 flex items-center gap-2">
            <Timer className="w-4 h-4 text-sky-400" /> Fastest training
          </div>
          <p className="text-2xl font-mono text-zinc-100 font-bold">{fastest.metrics.train_time_sec.toFixed(2)}s</p>
          <p className="text-sm text-zinc-400 mt-1">{fastest.name}</p>
        </div>
        <div className="glass-panel rounded-xl p-5">
          <div className="text-xs uppercase tracking-wider text-zinc-500 font-semibold mb-2 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-rose-400" /> Core answer
          </div>
          <p className="text-sm text-zinc-300 leading-relaxed">
            LLM-style tabular models are viable, but not yet a full replacement for tuned tree ensembles at scale.
          </p>
        </div>
      </div>

      <div className="glass-panel rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/[0.08]">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Dataset Winners</h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-zinc-500 text-xs uppercase tracking-wider">
              <th className="text-left px-6 py-3 font-medium">Dataset</th>
              <th className="text-left px-6 py-3 font-medium">Top Model</th>
              <th className="text-right px-6 py-3 font-medium">AUC</th>
              <th className="text-right px-6 py-3 font-medium">Train (s)</th>
            </tr>
          </thead>
          <tbody>
            {winners.map((r) => (
              <tr key={r.dataset} className="border-t border-white/[0.04]">
                <td className="px-6 py-3.5 text-zinc-200 font-medium">{r.dataset}</td>
                <td className="px-6 py-3.5 text-zinc-300">{r.winner.name}</td>
                <td className="px-6 py-3.5 text-right font-mono text-zinc-100">{r.winner.metrics.auc_roc.toFixed(4)}</td>
                <td className="px-6 py-3.5 text-right font-mono text-zinc-400">{r.winner.metrics.train_time_sec.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="glass-panel rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/[0.08]">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Family-Level Summary</h3>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-zinc-500 text-xs uppercase tracking-wider">
              <th className="text-left px-6 py-3 font-medium">Model Family</th>
              <th className="text-right px-6 py-3 font-medium">Avg AUC (Churn)</th>
              <th className="text-right px-6 py-3 font-medium">Avg AUC (Credit)</th>
              <th className="text-right px-6 py-3 font-medium">Avg AUC (Fraud)</th>
              <th className="text-left px-6 py-3 font-medium">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {catRows.map((row) => (
              <tr key={row.key} className="border-t border-white/[0.04]">
                <td className="px-6 py-3.5 text-zinc-100">{row.name}</td>
                <td className="px-6 py-3.5 text-right font-mono text-zinc-300">
                  {categoryAverage(churnData.models, row.key).toFixed(4)}
                </td>
                <td className="px-6 py-3.5 text-right font-mono text-zinc-300">
                  {categoryAverage(creditData.models, row.key).toFixed(4)}
                </td>
                <td className="px-6 py-3.5 text-right font-mono text-zinc-300">
                  {categoryAverage(fraudData.models, row.key).toFixed(4)}
                </td>
                <td className="px-6 py-3.5 text-zinc-400">{row.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  )
}
