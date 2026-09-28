---
paths:
  - "apps/**/*.ts"
  - "apps/**/*.tsx"
  - "ui-tui/**/*.ts"
  - "ui-tui/**/*.tsx"
  - "web/**/*.ts"
  - "web/**/*.tsx"
  - "website/**/*.ts"
  - "website/**/*.tsx"
  - "tests/**/*.ts"
  - "tests/**/*.tsx"
  - "package.json"
  - "package-lock.json"
  - "tsconfig*.json"
---

## TypeScript Style

Applies to TypeScript across Hermes: desktop, TUI, website, and future TS packages.

- Prefer small nanostores over component state when state is shared, reused, or read by distant UI.
- Let each feature own its atoms. Chat state belongs near chat, shell state near shell, shared state in `src/store`.
- Components that render from an atom should use `useStore`. Non-rendering actions should read with `$atom.get()`.
- Do not pass state through three components when the leaf can subscribe to the atom.
- Keep persistence beside the atom that owns it.
- Keep route roots thin. They compose routes and shell; they should not become controllers.
- No monolithic hooks. A hook should own one narrow job.
- Prefer colocated action modules over hidden god hooks.
- If a callback is pure side effect, use the terse void form:
  `onState={st => void setGatewayState(st)}`.
- Async UI handlers should make intent explicit:
  `onClick={() => void save()}`.
- Prefer interfaces for public props and shared object shapes. Avoid `type X = { ... }` for object props.
- Extend React primitives for props: `React.ComponentProps<'button'>`, `React.ComponentProps<typeof Dialog>`, `Omit<...>`, `Pick<...>`.
- Table-driven beats condition ladders when mapping ids, routes, or views.
- `src/app` owns routes, pages, and page-specific components.
- `src/store` owns shared atoms.
- `src/lib` owns shared pure helpers.
