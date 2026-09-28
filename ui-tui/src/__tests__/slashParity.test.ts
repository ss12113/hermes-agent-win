import { execFileSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { parseSlashCommand } from '@hermes/shared/slash'
import { describe, expect, it } from 'vitest'

import { findSlashCommand, SLASH_COMMANDS } from '../app/slash/registry.js'

type CommandRoute = 'fallback' | 'local' | 'native'

interface CommandRegistryLoad {
  error?: string
  names: string[]
}

const NATIVE_MUTATING_COMMANDS = new Set(['browser', 'busy', 'fast', 'reload-mcp', 'rollback', 'stop'])

const MUTATING_COMMANDS = [
  'bg',
  'btw',
  'branch',
  'browser',
  'busy',
  'clear',
  'compress',
  'fast',
  'model',
  'new',
  'personality',
  'queue',
  'reasoning',
  'reload-mcp',
  'retry',
  'rollback',
  'steer',
  'stop',
  'title',
  'tools',
  'undo',
  'verbose',
  'voice',
  'yolo'
] as const

const loadCommandRegistryNames = (): CommandRegistryLoad => {
  const here = dirname(fileURLToPath(import.meta.url))

  try {
    const names = JSON.parse(
      execFileSync(
        process.env.PYTHON ?? 'python3',
        [
          '-c',
          'import json; from hermes_cli.commands import COMMAND_REGISTRY; print(json.dumps([c.name for c in COMMAND_REGISTRY]))'
        ],
        { cwd: resolve(here, '../../..'), encoding: 'utf8' }
      )
    ) as string[]

    return { names: [...new Set(names)] }
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      names: []
    }
  }
}

const commandRegistry = loadCommandRegistryNames()
const registryIt = commandRegistry.error ? it.skip : it
const skipReason = commandRegistry.error ? commandRegistry.error.split('\n')[0] : ''

const LOCAL_COMMAND_NAMES = new Set(
  SLASH_COMMANDS.flatMap(command => [command.name, ...(command.aliases ?? [])].map(name => name.toLowerCase()))
)

const classifyRoute = (name: string): CommandRoute => {
  const normalized = name.toLowerCase()

  if (NATIVE_MUTATING_COMMANDS.has(normalized)) {
    return 'native'
  }

  if (LOCAL_COMMAND_NAMES.has(normalized)) {
    return 'local'
  }

  return 'fallback'
}

describe('slash parity matrix', () => {
  if (commandRegistry.error) {
    it.skip(`Python command registry unavailable: ${skipReason}`, () => {})
  }

  registryIt('classifies each command registry command as local/native/fallback', () => {
    const routes = Object.fromEntries(commandRegistry.names.map(name => [name, classifyRoute(name)]))

    expect(routes['model']).toBe('local')
    expect(routes['browser']).toBe('native')
    expect(routes['output-style']).toBe('fallback')
    expect(routes['reload-mcp']).toBe('native')
    expect(routes['rollback']).toBe('native')
    expect(routes['stop']).toBe('native')
  })

  registryIt('keeps every mutating command off slash-worker fallback', () => {
    const routes = Object.fromEntries(commandRegistry.names.map(name => [name, classifyRoute(name)]))

    for (const name of MUTATING_COMMANDS) {
      expect(routes[name], `missing command in registry: ${name}`).toBeDefined()
      expect(routes[name], `mutating command must not fallback: ${name}`).not.toBe('fallback')
    }
  })

  it('/q alias resolves to queue, not quit (#31983)', () => {
    // Regression for #31983: the TUI `quit` command used to carry alias `q`,
    // which collided with the Python-side `/queue` alias. TUI-local commands
    // dispatch before the backend, so `/q` resolved to /quit (session.die)
    // instead of queueing a prompt.
    const cmd = findSlashCommand('q')
    expect(cmd, '/q must resolve to a command').toBeDefined()
    expect(cmd!.name).toBe('queue')
  })

  it('leaves /compact for backend compression and keeps /compact-ui local', () => {
    expect(findSlashCommand('compact')).toBeUndefined()
    const cmd = findSlashCommand('compact-ui')
    expect(cmd, '/compact-ui must resolve to a local UI command').toBeDefined()
    expect(cmd!.name).toBe('compact-ui')
  })

  it('maps Claude-style theme/color aliases to the local skin command', () => {
    expect(findSlashCommand('theme')?.name).toBe('skin')
    expect(findSlashCommand('color')?.name).toBe('skin')
  })

  it('maps underscore terminal setup alias to the local terminal setup command', () => {
    expect(findSlashCommand('terminal_setup')?.name).toBe('terminal-setup')
  })

  it('maps Claude-style cost/rename aliases to existing usage/title commands', () => {
    expect(findSlashCommand('cost')?.name).toBe('usage')
    expect(findSlashCommand('rename')?.name).toBe('title')
  })

  it('keeps /diff local so it can use the TUI session workspace', () => {
    expect(findSlashCommand('diff')?.name).toBe('diff')
  })

  it('keeps /export local so it can write the live TUI transcript', () => {
    expect(findSlashCommand('export')?.name).toBe('export')
  })

  it('leaves sandbox status commands to the backend catalog instead of local TUI handlers', () => {
    expect(findSlashCommand('sandbox')).toBeUndefined()
    expect(findSlashCommand('sandbox-toggle')).toBeUndefined()
    expect(findSlashCommand('sandbox_toggle')).toBeUndefined()
  })

  it('/s alias resolves to steer, not sessions or a TUI-local command (#119176)', () => {
    // Same one-letter pattern as /q: the TUI-local registry must not shadow
    // the backend alias with a prefix command (/sessions) or its own binding.
    const cmd = findSlashCommand('s')
    expect(cmd, '/s must resolve to a command').toBeDefined()
    expect(cmd!.name).toBe('steer')
  })
})

describe('parseSlashCommand argument fidelity', () => {
  it('keeps a multi-line argument byte-for-byte', () => {
    const arg = 'first line\nsecond line\n\n  indented tail'

    expect(parseSlashCommand(`/pr-triage ${arg}`)).toEqual({
      arg,
      name: 'pr-triage'
    })
  })

  it('preserves runs of spaces inside the argument', () => {
    expect(parseSlashCommand('/goal ship   it').arg).toBe('ship   it')
  })

  it('still splits the command name off a single separator', () => {
    expect(parseSlashCommand('/cron add daily')).toEqual({
      arg: 'add daily',
      name: 'cron'
    })
    expect(parseSlashCommand('/exit')).toEqual({ arg: '', name: 'exit' })
    expect(parseSlashCommand('/exit ')).toEqual({ arg: '', name: 'exit' })
  })
})
