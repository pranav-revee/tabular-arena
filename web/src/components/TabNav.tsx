import { motion } from 'framer-motion'
import type { ComponentType } from 'react'
import {
  Landmark,
  LayoutDashboard,
  Radar,
  Scale,
  ShieldAlert,
} from 'lucide-react'

type Tab = {
  id: string
  label: string
  icon: ComponentType<{ className?: string }>
}

const tabs: Tab[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: LayoutDashboard,
  },
  {
    id: 'churn',
    label: 'Churn',
    icon: Radar,
  },
  {
    id: 'credit-risk',
    label: 'Credit Risk',
    icon: Landmark,
  },
  {
    id: 'fraud',
    label: 'Fraud',
    icon: ShieldAlert,
  },
  {
    id: 'verdict',
    label: 'Verdict',
    icon: Scale,
  },
]

interface TabNavProps {
  activeTab: string
  onTabChange: (id: string) => void
}

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="flex justify-center w-full mb-9 md:mb-10">
      <div className="flex flex-wrap justify-center gap-1.5 bg-white/[0.03] p-1.5 rounded-2xl border border-white/[0.08] backdrop-blur-xl shadow-lg shadow-black/20">
        {tabs.map((tab) => {
          const Icon = tab.icon
          return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`${activeTab === tab.id ? 'text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'} relative rounded-xl px-3.5 py-2 text-sm font-medium transition focus-visible:outline-2`}
            style={{
              WebkitTapHighlightColor: 'transparent',
            }}
          >
            {activeTab === tab.id && (
              <motion.span
                layoutId="bubble"
                className="absolute inset-0 z-10 bg-white/[0.1] rounded-full border border-white/[0.05] shadow-sm"
                transition={{
                  type: 'spring',
                  bounce: 0.2,
                  duration: 0.45,
                }}
              />
            )}
            <span className="relative z-20 inline-flex items-center gap-1.5">
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </span>
          </button>
          )
        })}
      </div>
    </div>
  )
}
