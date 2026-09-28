import { useCallback } from 'react'

import { useTheme } from './context'

// Retired skin names land on the canonical Nous skin so old muscle memory works.
const ALIASES: Record<string, string> = {
  ares: 'ember',
  default: 'nous',
  gold: 'nous',
  hermes: 'nous',
  'nous-light': 'nous'
}

export function useSkinCommand() {
  const { availableThemes, setTheme, themeName } = useTheme()

  return useCallback(
    (rawArg: string, rawCommand = '/skin') => {
      const arg = rawArg.trim()
      const normalized = arg.toLowerCase()
      const command = rawCommand.trim().split(/\s+/, 1)[0] || '/skin'
      const displayCommand = command.startsWith('/') ? command : `/${command}`

      if (!availableThemes.length) {
        return 'No desktop themes are available.'
      }

      const activeIndex = Math.max(
        0,
        availableThemes.findIndex(t => t.name === themeName)
      )

      if (!arg || normalized === 'next') {
        const next = availableThemes[(activeIndex + 1) % availableThemes.length]
        setTheme(next.name)

        return `Desktop theme switched to ${next.label}.`
      }

      if (normalized === 'list' || normalized === 'ls' || normalized === 'status') {
        const rows = availableThemes.map(t => `${t.name === themeName ? '*' : ' '} ${t.name.padEnd(10)} ${t.label}`)
        const aliasNote = displayCommand === '/skin' ? '' : `${displayCommand} is a compatibility alias for /skin.\n`

        return ['Desktop themes:', ...rows, '', `${aliasNote}Use ${displayCommand} <name>, or ${displayCommand} to cycle.`].join('\n')
      }

      const targetName = ALIASES[normalized] || normalized

      const target = availableThemes.find(
        t => t.name.toLowerCase() === targetName || t.label.toLowerCase() === normalized
      )

      if (!target) {
        return `Unknown desktop theme: ${arg}\nAvailable: ${availableThemes.map(t => t.name).join(', ')}`
      }

      setTheme(target.name)

      return `Desktop theme switched to ${target.label}.`
    },
    [availableThemes, setTheme, themeName]
  )
}
