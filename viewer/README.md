# Biomarker Results Viewer

A static React + Vite site for browsing the `biomarker_agent`'s per-compound
interpretation reports. Search compounds by DepMap/BRD id, drug name, or
gene/feature (refit-selected, baseline-top, or hypothesis-discussed), and view
each compound's full report plus the agent's run trace.

## Build the data bundle

The viewer reads a generated bundle under `public/data/` (gitignored). Generate
it from the agent's outputs:

    python viewer/scripts/build_data.py \
        --results data/interpretation_results \
        --out viewer/public/data

This writes `index.json`, one `<compound>.json` per compound, and copies each
compound's figures.

## Develop

    cd viewer
    npm install
    npm run dev        # dev server with hot reload

## Test

    cd viewer
    npm run test       # Vitest (search + component tests)

## Production build

    cd viewer
    npm run build      # type-check + bundle to viewer/dist/
    npm run preview    # serve the built site locally

Because the app uses HashRouter and a relative base, `viewer/dist/` can be served
from any static host (or opened via `python -m http.server -d dist`).

## Regenerate after a new agent run

Re-run `build_data.py`, then `npm run build` (or just refresh `npm run dev`).
