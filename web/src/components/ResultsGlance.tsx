import { motion } from 'framer-motion'
import { churnData } from '../data/churn'
import { creditData } from '../data/credit'

function bestByAuc(models: typeof churnData.models) {
  return [...models].sort((a, b) => b.metrics.auc_roc - a.metrics.auc_roc)[0]
}

function fastest(models: typeof churnData.models) {
  return [...models].sort(
    (a, b) => a.metrics.train_time_sec - b.metrics.train_time_sec
  )[0]
}

export function ResultsGlance() {
  const churnBest = bestByAuc(churnData.models)
  const creditBest = bestByAuc(creditData.models)
  const globalFastest = fastest([...churnData.models, ...creditData.models])
  const churnFoundation = churnData.models.find((m) => m.name === 'TabPFN')
  const creditFoundation = creditData.models.find((m) => m.name === 'TabPFN')
  const foundationDelta =
    churnFoundation && creditFoundation
      ? (churnFoundation.metrics.auc_roc - creditFoundation.metrics.auc_roc).toFixed(4)
      : '—'

  const cards = [
    {
      label: 'BEST AUC (CHURN)',
      value: churnBest.metrics.auc_roc.toFixed(4),
      sub: churnBest.name,
      color: 'border-l-amber-500',
      delay: 0.3,
    },
    {
      label: 'BEST AUC (CREDIT)',
      value: creditBest.metrics.auc_roc.toFixed(4),
      sub: creditBest.name,
      color: 'border-l-violet-500',
      delay: 0.35,
    },
    {
      label: 'FASTEST',
      value: `${globalFastest.metrics.train_time_sec.toFixed(2)}s`,
      sub: globalFastest.name,
      color: 'border-l-indigo-500',
      delay: 0.4,
    },
    {
      label: 'FOUNDATION DROP (SMALL → LARGE)',
      value: foundationDelta,
      sub: 'TabPFN AUC delta',
      color: 'border-l-rose-500',
      delay: 0.45,
    },
  ]
  return (
    <section className="mb-12 md:mb-14">
      <h2 className="text-lg font-semibold text-zinc-100 mb-2">Results at a Glance</h2>
      <p className="text-sm text-zinc-400 mb-6">
        One-line view of how the narrative changes across dataset scale.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((card, index) => (
          <motion.div
            key={index}
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
              delay: card.delay,
            }}
            className={`glass-panel border-l-4 ${card.color} rounded-r-xl rounded-l-sm p-6 hover:bg-white/[0.06] transition-colors duration-300`}
          >
            <div className="flex flex-col h-full justify-between">
              <div>
                <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider block mb-2">
                  {card.label}
                </span>
                <span className="text-3xl font-bold text-zinc-100 font-mono block mb-1 drop-shadow-md">
                  {card.value}
                </span>
              </div>
              <span className="text-sm text-zinc-400 mt-2">{card.sub}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  )
}
