# AI Frontier

An interactive web dashboard comparing 100+ large language models across cost, speed, and quality metrics.

## Stack

- **Next.js 16** — App Router, TypeScript
- **Tailwind CSS v4** — utility-first styling
- **D3.js + Recharts** — interactive data visualization
- **Playwright** — autonomous visual testing via MCP

## Getting Started

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project Structure

```
src/
└── app/          # Next.js App Router pages and layouts
public/           # Static assets
.claude/          # Claude Code project settings
.mcp.json         # Playwright MCP server config
```

## Scripts

| Command | Description |
|---|---|
| `pnpm dev` | Start development server |
| `pnpm build` | Production build |
| `pnpm lint` | Run ESLint |
