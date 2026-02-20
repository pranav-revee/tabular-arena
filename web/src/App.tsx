import { useState } from 'react'
import { TabNav } from './components/TabNav'
import { HeroSection } from './components/HeroSection'
import { BenchmarkSetup } from './components/BenchmarkSetup'
import { ResultsGlance } from './components/ResultsGlance'
import { Footer } from './components/Footer'
import { ChurnTab } from './components/ChurnTab'
import { CreditRiskTab } from './components/CreditRiskTab'
import { motion, AnimatePresence } from 'framer-motion'

export function App() {
  const [activeTab, setActiveTab] = useState('overview')
  return (
    <div className="min-h-screen w-full bg-[#0a0f1a] text-zinc-200 selection:bg-emerald-500/30 selection:text-emerald-200 font-sans relative overflow-hidden">
      {/* Ambient Background Orbs */}
      <div className="ambient-light ambient-emerald" />
      <div className="ambient-light ambient-violet" />
      <div className="ambient-light ambient-amber" />

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12 md:py-20">
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
