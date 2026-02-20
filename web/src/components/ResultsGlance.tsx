import { motion } from 'framer-motion'

export function ResultsGlance() {
  const cards = [
    {
      label: 'BEST AUC (CHURN)',
      value: '0.8445',
      sub: 'AutoGluon',
      color: 'border-l-amber-500',
      delay: 0.3,
    },
    {
      label: 'BEST AUC (CREDIT)',
      value: '0.7857',
      sub: 'AutoGluon',
      color: 'border-l-amber-500',
      delay: 0.35,
    },
    {
      label: 'FASTEST',
      value: '0.14s',
      sub: 'XGBoost (Default)',
      color: 'border-l-indigo-500',
      delay: 0.4,
    },
    {
      label: 'BEST DEFAULT',
      value: '0.8436',
      sub: 'CatBoost (Default)',
      color: 'border-l-emerald-500',
      delay: 0.45,
    },
  ]
  return (
    <section className="mb-16">
      <h2 className="text-lg font-semibold text-zinc-100 mb-6">
        Results at a Glance
      </h2>
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
            className={`bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] border-l-4 ${card.color} rounded-r-xl rounded-l-sm p-6 hover:bg-white/[0.04] transition-colors duration-300 shadow-lg shadow-black/10`}
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
