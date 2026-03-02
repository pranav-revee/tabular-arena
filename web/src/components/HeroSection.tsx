import { motion } from 'framer-motion'

export function HeroSection() {
  return (
    <section className="mb-12 md:mb-14 space-y-6">
      <motion.div
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
        }}
      >
        <div className="flex flex-wrap gap-2 mb-4">
          <span className="rounded-full border border-emerald-300/20 bg-emerald-300/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-200">
            Modern Tabular Benchmark
          </span>
          <span className="rounded-full border border-rose-300/20 bg-rose-300/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-rose-200">
            LLM vs Classical ML
          </span>
        </div>
        <h1 className="text-4xl md:text-6xl font-bold text-zinc-100 mb-4 tracking-tight leading-[1.05]">
          Tabular Arena <span className="text-teal-400 font-normal">⚔️</span>
        </h1>
        <p className="text-lg md:text-xl text-zinc-300 max-w-4xl leading-relaxed">
          If a foundation model can truly read tabular data, it should compete
          with tuned gradient boosting across small, medium, and large
          real-world datasets. This benchmark makes that comparison visible.
        </p>
      </motion.div>

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
          delay: 0.1,
        }}
        className="grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-5"
      >
        <div className="glass-panel rounded-2xl p-6 md:p-7 space-y-3">
          <p className="text-sm uppercase tracking-[0.16em] text-zinc-500 font-semibold">
            Story
          </p>
          <p className="text-zinc-300 leading-relaxed">
            Fundamental raised $255M for NEXUS and claims one model can replace
            the entire tabular ML pipeline. Since it is closed-source, we test
            the open alternatives under one protocol: same data splits, same
            metrics, same compute reporting.
          </p>
          <p className="text-zinc-300 leading-relaxed">
            We benchmark <strong className="text-rose-300">TabPFN</strong> and{' '}
            <strong className="text-sky-300">FT-Transformer</strong> against{' '}
            <strong className="text-emerald-300">LightGBM</strong>,{' '}
            <strong className="text-emerald-300">CatBoost</strong>,{' '}
            <strong className="text-indigo-300">XGBoost</strong>, and{' '}
            <strong className="text-amber-300">AutoGluon</strong>.
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-6 md:p-7 space-y-4">
          <p className="text-sm uppercase tracking-[0.16em] text-zinc-500 font-semibold">
            Thesis
          </p>
          <div className="space-y-3 text-sm text-zinc-300">
            <div className="rounded-xl border border-white/10 bg-black/20 p-3">
              <p className="text-zinc-100 font-semibold mb-1">Small data</p>
              Foundation models can be competitive quickly.
            </div>
            <div className="rounded-xl border border-white/10 bg-black/20 p-3">
              <p className="text-zinc-100 font-semibold mb-1">Large tabular</p>
              Tuned boosted trees usually remain strongest.
            </div>
            <div className="rounded-xl border border-white/10 bg-black/20 p-3">
              <p className="text-zinc-100 font-semibold mb-1">Production lens</p>
              Accuracy must be weighed against latency + memory.
            </div>
          </div>
        </div>
      </motion.section>
    </section>
  )
}
