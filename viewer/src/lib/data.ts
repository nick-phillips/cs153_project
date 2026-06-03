/// <reference types="vite/client" />
import type { CompoundData, IndexEntry } from './types';

// BASE_URL is './' (see vite.config base), so these resolve relative to the
// served index.html — which is constant under HashRouter.
const BASE = import.meta.env.BASE_URL;

export async function loadIndex(): Promise<IndexEntry[]> {
  const res = await fetch(`${BASE}data/index.json`);
  if (!res.ok) throw new Error(`Failed to load index.json (HTTP ${res.status})`);
  return res.json();
}

export async function loadCompound(id: string): Promise<CompoundData> {
  const res = await fetch(`${BASE}data/${id}.json`);
  if (!res.ok) throw new Error(`Failed to load compound "${id}" (HTTP ${res.status})`);
  return res.json();
}

export function figureUrl(id: string, path: string): string {
  return `${BASE}data/${id}/${path}`;
}
