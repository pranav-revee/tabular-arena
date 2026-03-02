import { motion } from 'framer-motion'
import { Building2, Database, ShieldAlert } from 'lucide-react'

export function BenchmarkSetup() {
  return (
    <motion.section
      initial={{
        opacity: 0,
        y: 20,
      }}
      animate={{
        opacity: 1,
        y: 0,
      }}
      transition={{
        duration: 0.5,
        delay: 0.2,
      }}
      className="space-y-6 mb-12 md:mb-14"
    >
      <div className="glass-panel rounded-2xl p-6">
        <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-4">
          Datasets (Progressive Difficulty)
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-xl border border-white/10 bg-black/20 p-4">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-emerald-300" />
              <span className="text-zinc-100 font-semibold">Churn</span>
            </div>
            <p className="text-zinc-400 text-sm font-mono">
              7,043 rows · 19 features · 26% positive
            </p>
            <p className="text-zinc-500 text-xs mt-2">
              Small mixed-feature business tabular.
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-black/20 p-4">
            <div className="flex items-center gap-2 mb-2">
              <Building2 className="w-4 h-4 text-indigo-300" />
              <span className="text-zinc-100 font-semibold">Credit Risk</span>
            </div>
            <p className="text-zinc-400 text-sm font-mono">
              307,511 rows · 302 features · 8% positive
            </p>
            <p className="text-zinc-500 text-xs mt-2">
              Large engineered multi-table credit features.
            </p>
          </div>

          <div className="rounded-xl border border-white/10 bg-black/20 p-4">
            <div className="flex items-center gap-2 mb-2">
              <ShieldAlert className="w-4 h-4 text-rose-300" />
              <span className="text-zinc-100 font-semibold">Fraud</span>
            </div>
            <p className="text-zinc-400 text-sm font-mono">
              590,540 rows · 434 features · 3.5% positive
            </p>
            <p className="text-zinc-500 text-xs mt-2">
              Very large imbalanced transaction classification.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-6">
        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-5">
            Model Positioning
          </h3>
          <div className="space-y-3 text-sm">
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-3 py-2">
              <span className="text-emerald-300 font-semibold">
                Gradient Boosting:
              </span>{' '}
              <span className="text-zinc-300">
                LightGBM, CatBoost, XGBoost — strongest large-scale tabular
                baseline.
              </span>
            </div>
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2">
              <span className="text-amber-300 font-semibold">AutoML:</span>{' '}
              <span className="text-zinc-300">
                AutoGluon — high-accuracy ensemble, higher runtime and memory.
              </span>
            </div>
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 py-2">
              <span className="text-rose-300 font-semibold">
                Foundation Model:
              </span>{' '}
              <span className="text-zinc-300">
                TabPFN — strongest in lower-data settings, limited at very large
                scale.
              </span>
            </div>
            <div className="rounded-xl border border-sky-500/20 bg-sky-500/10 px-3 py-2">
              <span className="text-sky-300 font-semibold">Deep Learning:</span>{' '}
              <span className="text-zinc-300">
                FT-Transformer — benefits from data + tuning, but not
                consistently dominant.
              </span>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-wider mb-4">
            Protocol
          </h3>
          <ul className="space-y-2 text-sm text-zinc-300">
            <li>• 5-fold stratified CV</li>
            <li>• 80/20 holdout split (seed 42)</li>
            <li>• AUC-ROC primary ranking metric</li>
            <li>• log loss, train time, memory, inference latency</li>
            <li>• scaling curves by sample size</li>
          </ul>
        </div>
      </div>
    </motion.section>
  )
}
