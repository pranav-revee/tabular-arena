
export function Footer() {
  return (
    <footer className="pt-6 pb-14">
      <div className="glass-panel rounded-2xl px-5 py-5 md:px-6 md:py-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <p className="text-sm text-zinc-300 font-medium">
              Built by Pranav Reveendran
            </p>
            <p className="text-xs text-zinc-500 mt-1">
              Tabular benchmark narrative: foundation vs classical ML.
            </p>
          </div>
          <p className="text-xs font-mono text-zinc-500">
            Apple M3 Pro · 18GB RAM · Darwin 25.2.0 · Python 3.12.9
          </p>
        </div>
      </div>
    </footer>
  )
}
