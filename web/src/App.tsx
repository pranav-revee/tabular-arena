import { useState } from 'react'
import { TabNav } from './components/TabNav'
import { HeroSection } from './components/HeroSection'
import { BenchmarkSetup } from './components/BenchmarkSetup'
import { ResultsGlance } from './components/ResultsGlance'
import { Footer } from './components/Footer'
import { ChurnTab } from './components/ChurnTab'
import { CreditRiskTab } from './components/CreditRiskTab'
import { FraudTab } from './components/FraudTab'
import { VerdictTab } from './components/VerdictTab'
import { motion, AnimatePresence } from 'framer-motion'

export function App() {
  const [activeTab, setActiveTab] = useState('overview')
  return (
    <div className="min-h-screen w-full text-zinc-200 selection:bg-emerald-500/30 selection:text-emerald-200 font-sans relative overflow-hidden">
      <div className="grid-overlay" />
      <div className="ambient-light ambient-emerald" />
      <div className="ambient-light ambient-violet" />
      <div className="ambient-light ambient-amber" />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-10 md:py-14">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-6 md:mb-8"
        >
          <div className="glass-panel rounded-2xl px-4 py-3 md:px-5 md:py-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div className="text-xs md:text-sm text-zinc-300 tracking-wide">
              <span className="inline-flex items-center rounded-full border border-violet-300/20 bg-violet-300/10 px-2 py-0.5 text-[11px] font-semibold uppercase text-violet-200 mr-2">
                Research Story
              </span>
              Can foundation-style models read tabular data as well as tuned trees?
            </div>
            <div className="text-[11px] md:text-xs font-mono text-zinc-400">
              3 datasets · 9 models · unified protocol
            </div>
          </div>
        </motion.div>

        <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

        <main className="min-h-[60vh]">
          <AnimatePresence mode="wait">
            {activeTab === 'overview' ? (
              <motion.div
                key="overview"
                initial={{
                  opacity: 0,
                  x: -20,
                }}
                animate={{
                  opacity: 1,
                  x: 0,
                }}
                exit={{
                  opacity: 0,
                  x: 20,
                }}
                transition={{
                  duration: 0.3,
                }}
              >
                <HeroSection />
                <BenchmarkSetup />
                <ResultsGlance />
              </motion.div>
            ) : activeTab === 'churn' ? (
              <ChurnTab key="churn" />
            ) : activeTab === 'credit-risk' ? (
              <CreditRiskTab key="credit-risk" />
            ) : activeTab === 'fraud' ? (
              <FraudTab key="fraud" />
            ) : activeTab === 'verdict' ? (
              <VerdictTab key="verdict" />
            ) : (
              <motion.div
                key="placeholder"
                initial={{
                  opacity: 0,
                  scale: 0.95,
                }}
                animate={{
                  opacity: 1,
                  scale: 1,
                }}
                exit={{
                  opacity: 0,
                  scale: 0.95,
                }}
                transition={{
                  duration: 0.3,
                }}
                className="flex flex-col items-center justify-center h-96 text-center border border-dashed border-white/[0.08] rounded-2xl bg-white/[0.02] backdrop-blur-xl"
              >
                <div className="p-4 rounded-full bg-white/[0.05] mb-4 border border-white/[0.05]">
                  <span className="text-2xl">🚧</span>
                </div>
                <h3 className="text-xl font-semibold text-zinc-200 mb-2">
                  {activeTab.charAt(0).toUpperCase() +
                    activeTab.slice(1).replace('-', ' ')}
                </h3>
                <p className="text-zinc-500 max-w-md">
                  Detailed benchmark results and analysis for this section are
                  coming soon.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        <Footer />
      </div>
    </div>
  )
}
