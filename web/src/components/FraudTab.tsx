import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Clock3, Database, HardDrive, Trophy } from 'lucide-react'
import { fraudData, FRAUD_MODEL_COLORS } from '../data/fraud'
import { CATEGORY_COLORS, CATEGORY_LABELS } from '../data/churn'

function fmtInference(msPer1k: number) {
  return msPer1k >= 1000 ? `${(msPer1k / 1000).toFixed(1)}s` : `${msPer1k.toFixed(1)}`
}

export function FraudTab() {
  const sortedModels = useMemo(
    () => [...fraudData.models].sort((a, b) => b.metrics.auc_roc - a.metrics.auc_roc),
    []
  )

  const best = sortedModels[0]
  const fastest = [...fraudData.models].sort(
    (a, b) => a.metrics.train_time_sec - b.metrics.train_time_sec
  )[0]
  const lightest = [...fraudData.models].sort(
    (a, b) => a.metrics.peak_memory_mb - b.metrics.peak_memory_mb
  )[0]

  const fm = sortedModels.find((m) => m.category === 'foundation_model')
  const bestTree = sortedModels.find((m) => m.category === 'gradient_boosting')
  const gap = fm && bestTree ? (bestTree.metrics.auc_roc - fm.metrics.auc_roc).toFixed(4) : '—'

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      className="space-y-8"
    >
      <div className="glass-panel rounded-2xl p-5 md:p-6">
        <div className="flex items-start gap-3 md:gap-4">
          <div className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08]">
            <Database className="w-5 h-5 text-rose-300" />
          </div>
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-zinc-100 tracking-tight">
              IEEE-CIS Fraud Detection
            </h2>
            <p className="text-zinc-500 mt-1 font-mono text-sm">
              {fraudData.n_samples.toLocaleString()} rows · {fraudData.n_features} features ·{' '}
              {(fraudData.target_rate * 100).toFixed(1)}% positive · Kaggle Competition
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel rounded-xl p-4 border-l-4 border-l-amber-500">
          <div className="flex items-center gap-2 mb-2 text-zinc-500 text-xs uppercase tracking-wider font-semibold">
            <Trophy className="w-4 h-4 text-amber-400" /> Best AUC
          </div>
          <p className="text-2xl font-mono font-bold text-zinc-100">{best.metrics.auc_roc.toFixed(4)}</p>
          <p className="text-sm text-amber-300">{best.name}</p>
        </div>
        <div className="glass-panel rounded-xl p-4 border-l-4 border-l-emerald-500">
          <div className="flex items-center gap-2 mb-2 text-zinc-500 text-xs uppercase tracking-wider font-semibold">
            <Clock3 className="w-4 h-4 text-emerald-400" /> Fastest Train
          </div>
          <p className="text-2xl font-mono font-bold text-zinc-100">{fastest.metrics.train_time_sec.toFixed(1)}s</p>
          <p className="text-sm text-emerald-300">{fastest.name}</p>
        </div>
        <div className="glass-panel rounded-xl p-4 border-l-4 border-l-sky-500">
          <div className="flex items-center gap-2 mb-2 text-zinc-500 text-xs uppercase tracking-wider font-semibold">
            <HardDrive className="w-4 h-4 text-sky-400" /> Lightest Memory
          </div>
          <p className="text-2xl font-mono font-bold text-zinc-100">{lightest.metrics.peak_memory_mb.toFixed(0)} MB</p>
          <p className="text-sm text-sky-300">{lightest.name}</p>
        </div>
        <div className="glass-panel rounded-xl p-4 border-l-4 border-l-rose-500">
          <div className="flex items-center gap-2 mb-2 text-zinc-500 text-xs uppercase tracking-wider font-semibold">
            <AlertTriangle className="w-4 h-4 text-rose-400" /> FM Gap
          </div>
          <p className="text-2xl font-mono font-bold text-zinc-100">{gap}</p>
          <p className="text-sm text-rose-300">Best tree vs TabPFN AUC</p>
        </div>
      </div>

      <div className="glass-panel rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-white/[0.08]">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Fraud Leaderboard</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-zinc-500 text-xs uppercase tracking-wider">
                <th className="text-left px-5 py-3 font-medium">#</th>
                <th className="text-left px-5 py-3 font-medium">Model</th>
                <th className="text-left px-5 py-3 font-medium">Type</th>
                <th className="text-right px-5 py-3 font-medium">AUC</th>
                <th className="text-right px-5 py-3 font-medium">Log Loss</th>
                <th className="text-right px-5 py-3 font-medium">Train (s)</th>
                <th className="text-right px-5 py-3 font-medium">Inf (ms/1k)</th>
              </tr>
            </thead>
            <tbody>
              {sortedModels.map((m, i) => (
                <tr key={m.name} className="border-t border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="px-5 py-3.5 text-zinc-500 font-mono">{i + 1}</td>
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: FRAUD_MODEL_COLORS[m.name] }} />
                      <span className="text-zinc-200 font-medium">{m.name}</span>
                    </div>
                  </td>
                  <td className="px-5 py-3.5">
                    <span className={`text-xs px-2 py-0.5 rounded-full border ${CATEGORY_COLORS[m.category]}`}>
                      {CATEGORY_LABELS[m.category]}
                    </span>
                  </td>
                  <td className="px-5 py-3.5 text-right font-mono text-zinc-100">{m.metrics.auc_roc.toFixed(4)}</td>
                  <td className="px-5 py-3.5 text-right font-mono text-zinc-400">{m.metrics.log_loss.toFixed(4)}</td>
                  <td className="px-5 py-3.5 text-right font-mono text-zinc-400">{m.metrics.train_time_sec.toFixed(1)}</td>
                  <td className="px-5 py-3.5 text-right font-mono text-zinc-400">
                    {fmtInference(m.metrics.inference_time_ms_per_1k)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-6 space-y-3">
        <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Takeaways</h3>
        <p className="text-zinc-300 text-sm leading-relaxed">
          On large, imbalanced fraud data, tuned gradient boosting remains the strongest frontier.
          Foundation-model performance is competitive but trails top tuned trees by a visible margin.
        </p>
        <p className="text-zinc-400 text-sm leading-relaxed">
          AutoML matches top AUC but at significantly higher runtime and memory cost.
        </p>
      </div>
    </motion.div>
  )
}
