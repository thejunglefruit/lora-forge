// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 The Jungle Fruit
import { useEffect, useRef, useState } from 'react'

// LoRA Forge: a thin UI over the API (forge_web/api.py), which in turn calls the forge_core library.
// Steps: Project → Dataset (ingest + validation) → Captions (auto + editing) → Config (estimate) → Train → Run note / pick.

type Project = { name: string; folder: string; type: string; model: string; images: number }
type Detail = {
  name: string; type: string; model: string; trigger: string | null; eval_prompts: string[]
  images: string[]; manifest: { images?: { path?: string; flags?: string[]; width?: number; height?: number; blur_var?: number; faces?: number | null }[]; meta?: { summary?: Record<string, unknown> } }
  captions: Record<string, string | null>; runs: Record<string, unknown>[]; doctrine: { include: string[]; omit: string[]; instruction: string; caption_engine: string }
  requirements: Record<string, unknown>; license: Record<string, unknown>; dataset_report: string | null
}
type Health = { real_captioner: boolean; aitoolkit_backend: boolean; gpu: string | null }
type Models = { models: { key: string; arch: string; status: string; caption_engine: string }[]; types: string[] }

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) throw new Error((await r.text()) || r.statusText)
  return r.json()
}
const json = (b: unknown, method = 'POST'): RequestInit => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) })
const base = (name: string) => `/api/projects/${encodeURIComponent(name)}`

export default function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [models, setModels] = useState<Models | null>(null)
  const [current, setCurrent] = useState<string | null>(null)
  const [detail, setDetail] = useState<Detail | null>(null)
  const [step, setStep] = useState<'dataset' | 'captions' | 'config' | 'train'>('dataset')
  const [message, setMessage] = useState('')
  const [draft, setDraft] = useState({ name: '', type: 'character', model: 'flux', trigger: '' })

  const reload = async () => {
    setProjects(await api('/api/projects'))
    if (current) setDetail(await api(base(current)))
  }
  useEffect(() => {
    void api<Health>('/api/health').then(setHealth)
    void api<Models>('/api/models').then(setModels)
    void api<Project[]>('/api/projects').then((ps) => { setProjects(ps); if (ps.length) setCurrent(ps[0].folder) })
  }, [])
  useEffect(() => {
    if (current) void api<Detail>(base(current)).then(setDetail)
  }, [current])

  const notify = (m: string) => { setMessage(m); setTimeout(() => setMessage(''), 6000) }
  const create = async () => {
    try {
      const r = await api<{ folder: string; trigger: string }>('/api/projects', json(draft))
      notify(`Project created. Trigger: ${r.trigger ?? '-'}`)
      setDraft({ ...draft, name: '' })
      await api<Project[]>('/api/projects').then(setProjects)
      setCurrent(r.folder)
    } catch (e) { notify(String(e)) }
  }

  return (
    <div className="app">
      <header>
        <b>LoRA Forge</b>
        <span className="muted">dataset + LoRA training orchestrator</span>
        <span className="spacer" />
        {health && <span className={health.real_captioner ? 'ok' : 'muted'}>real captioner (Ollama): {health.real_captioner ? 'yes' : 'no'}</span>}
        {health && <span className={health.aitoolkit_backend ? 'ok' : 'muted'}>ai-toolkit: {health.aitoolkit_backend ? 'yes' : 'no (mock)'}</span>}
        {message && <span className="notice">{message}</span>}
      </header>
      <aside>
        <div className="section">Projects</div>
        <ul className="list">
          {projects.map((p) => (
            <li key={p.folder} className={p.folder === current ? 'sel' : ''} onClick={() => setCurrent(p.folder)}>{p.name} <small>{p.type} · {p.model} · {p.images} img</small></li>
          ))}
        </ul>
        <div className="section">New project</div>
        <div className="form">
          <input placeholder="name (e.g. ceramic-mug)" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          <select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value })}>{models?.types.map((t) => <option key={t}>{t}</option>)}</select>
          <select value={draft.model} onChange={(e) => setDraft({ ...draft, model: e.target.value })}>{models?.models.map((m) => <option key={m.key} value={m.key}>{m.key} ({m.arch}, {m.caption_engine}{m.status !== 'active' ? ', unverified' : ''})</option>)}</select>
          <input placeholder="trigger (optional)" value={draft.trigger} onChange={(e) => setDraft({ ...draft, trigger: e.target.value })} />
          <button className="primary" onClick={() => void create()}>Create</button>
        </div>
      </aside>
      <main>
        {!detail ? <div className="muted">Pick or create a project.</div> : (
          <>
            <div className="head">
              <h2>{detail.name}</h2>
              <span className="muted">{detail.type} · {detail.model} · trigger <b>{detail.trigger ?? '-'}</b> · {detail.images.length} images · licence {String(detail.license?.id ?? '?')} {detail.license?.commercial_use === false ? '(NON-commercial)' : ''}</span>
            </div>
            <nav className="steps">
              {(['dataset', 'captions', 'config', 'train'] as const).map((p, i) => (
                <button key={p} className={step === p ? 'active' : ''} onClick={() => setStep(p)}>{i + 1}. {{ dataset: 'Dataset', captions: 'Captions', config: 'Config & estimate', train: 'Train & run notes' }[p]}</button>
              ))}
            </nav>
            {step === 'dataset' && <Dataset d={detail} notify={notify} reload={reload} />}
            {step === 'captions' && <Captions d={detail} notify={notify} reload={reload} />}
            {step === 'config' && <Config d={detail} notify={notify} reload={reload} />}
            {step === 'train' && <Train d={detail} notify={notify} reload={reload} />}
          </>
        )}
      </main>
    </div>
  )
}

type PP = { d: Detail; notify: (m: string) => void; reload: () => Promise<void> }

function Dataset({ d, notify, reload }: PP) {
  const [folder, setFolder] = useState('')
  const [busy, setBusy] = useState(false)
  const ingest = async (body: RequestInit) => {
    setBusy(true)
    try {
      const r = await api<{ n: number; blocked: boolean }>(`${base(d.name)}/ingest`, body)
      notify(`${r.n} images · ${r.blocked ? 'BLOCKED: review the flags' : 'no flags'}`)
      await reload()
    } catch (e) { notify(String(e)) } finally { setBusy(false) }
  }
  const upload = (fl: FileList | null) => {
    if (!fl?.length) return
    const fd = new FormData()
    for (const f of Array.from(fl)) fd.append('files', f)
    void ingest({ method: 'POST', body: fd })
  }
  const imgs = d.manifest?.images ?? []
  const info = (name: string) => imgs.find((i) => (i.path ?? '') === name)
  return (
    <div>
      <div className="row">
        <label className="button">Upload images…<input type="file" multiple hidden accept="image/*" onChange={(e) => upload(e.target.files)} /></label>
        <span>or a local folder:</span>
        <input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="/path/to/images" style={{ width: 320 }} />
        <button onClick={() => void ingest(json({ folder }))} disabled={busy}>Ingest and validate</button>
      </div>
      <div className="muted">Requirements for this model/type: {JSON.stringify(d.requirements)}</div>
      <div className="grid">
        {d.images.map((n) => {
          const i = info(n)
          return (
            <div key={n} className={`card ${i?.flags?.length ? 'flagged' : ''}`}>
              <img src={`${base(d.name)}/image/${encodeURIComponent(n)}`} alt={n} />
              <div className="filename">{n}</div>
              {i && <small>{i.width}×{i.height} · blur {Math.round(i.blur_var ?? 0)} · faces {i.faces ?? '-'}</small>}
              {i?.flags?.length ? <div className="flags">{i.flags.join(' · ')}</div> : null}
              <button className="mini" onClick={async () => { await fetch(`${base(d.name)}/image/${encodeURIComponent(n)}`, { method: 'DELETE' }); await reload() }}>remove</button>
            </div>
          )
        })}
      </div>
      {d.dataset_report && <details><summary>Dataset report</summary><pre>{d.dataset_report}</pre></details>}
    </div>
  )
}

function Captions({ d, notify, reload }: PP) {
  const [busy, setBusy] = useState(false)
  const generate = async (real: boolean, force: boolean) => {
    setBusy(true)
    try {
      const r = await api<{ written: number; kept: number; engine: string; seconds: number }>(`${base(d.name)}/captions`, json({ real, force }))
      notify(`${r.written} written, ${r.kept} kept · ${r.engine} · ${r.seconds} s`)
      await reload()
    } catch (e) { notify(String(e)) } finally { setBusy(false) }
  }
  const save = async (n: string, text: string) => {
    await fetch(`${base(d.name)}/caption/${encodeURIComponent(n)}`, json({ text }, 'PUT'))
  }
  return (
    <div>
      <div className="box">
        <b>Doctrine ({d.type}, engine {d.doctrine.caption_engine}):</b> describe {d.doctrine.include.join(', ')}; do <b>not</b> describe {d.doctrine.omit.join(', ')} (the trigger <b>{d.trigger ?? '-'}</b> learns it).
      </div>
      <div className="row">
        <button onClick={() => void generate(false, false)} disabled={busy}>Generate (mock)</button>
        <button className="primary" onClick={() => void generate(true, false)} disabled={busy}>Generate with a vision model (Ollama)</button>
        <button onClick={() => void generate(true, true)} disabled={busy}>Regenerate all (overwrites edits)</button>
        {busy && <span className="muted">working… (on CPU it takes ~30-60 s per image)</span>}
      </div>
      <div className="grid wide">
        {d.images.map((n) => (
          <div key={n} className="card">
            <img src={`${base(d.name)}/image/${encodeURIComponent(n)}`} alt={n} />
            <div className="filename">{n}</div>
            <textarea defaultValue={d.captions[n] ?? ''} rows={4} placeholder="(no caption)" onBlur={(e) => void save(n, e.target.value)} />
          </div>
        ))}
      </div>
    </div>
  )
}

function Config({ d, notify, reload }: PP) {
  const [gpu, setGpu] = useState('RTX 4090')
  const [vram, setVram] = useState(24)
  const [res, setRes] = useState<{ run_id: string; steps: number; config: string; estimate_text: string } | null>(null)
  const create = async () => {
    try {
      const r = await api<{ run_id: string; steps: number; config: string; estimate_text: string }>(`${base(d.name)}/config`, json({ gpu, gpu_vram: vram }))
      setRes(r)
      notify(`run ${r.run_id} · ${r.steps} steps`)
      await reload()
    } catch (e) { notify(String(e)) }
  }
  return (
    <div>
      <div className="row">
        <label>GPU</label>
        <select value={gpu} onChange={(e) => setGpu(e.target.value)}>{['RTX 4090', 'RTX 3090', 'A100 80GB', 'H100', 'L40S'].map((g) => <option key={g}>{g}</option>)}</select>
        <label>VRAM (GB)</label><input type="number" value={vram} onChange={(e) => setVram(Number(e.target.value))} style={{ width: 70 }} />
        <button className="primary" onClick={() => void create()}>Generate config and estimate</button>
      </div>
      {res && (
        <>
          <pre className="estimate">{res.estimate_text}</pre>
          <details><summary>ai-toolkit YAML ({res.run_id})</summary><pre>{res.config}</pre></details>
        </>
      )}
      <div className="muted">Evaluation prompts: {d.eval_prompts.join(' · ')}</div>
    </div>
  )
}

function Train({ d, notify, reload }: PP) {
  const [sel, setSel] = useState<string | null>(null)
  const [run, setRun] = useState<{ run: Record<string, unknown>; progress: Record<string, unknown> | null; log: string; report: string | null } | null>(null)
  const [checkpoint, setCheckpoint] = useState('')
  const timer = useRef<number | null>(null)
  const load = async (id: string) => setRun(await api(`${base(d.name)}/runs/${id}`))
  useEffect(() => {
    if (!sel) return
    void load(sel)
    timer.current = window.setInterval(() => void load(sel), 2000)
    return () => { if (timer.current) window.clearInterval(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel])
  const launch = async (mock: boolean) => {
    if (!sel) return
    try { await api(`${base(d.name)}/train/${sel}`, json({ mock })); notify(mock ? 'simulated run launched' : 'training launched') } catch (e) { notify(String(e)) }
  }
  const writeNote = async () => { if (!sel) return; try { await api(`${base(d.name)}/report/${sel}`, { method: 'POST' }); await load(sel); notify('run note written') } catch (e) { notify(String(e)) } }
  const pick = async () => { if (!sel) return; try { const r = await api<{ calibration: Record<string, unknown> }>(`${base(d.name)}/pick/${sel}`, json({ checkpoint })); notify(`pick saved · calibration ${JSON.stringify(r.calibration)}`); await reload() } catch (e) { notify(String(e)) } }
  return (
    <div className="two-col">
      <div>
        <div className="section">Runs</div>
        <ul className="list">
          {d.runs.map((r) => (
            <li key={String(r.run_id)} className={sel === r.run_id ? 'sel' : ''} onClick={() => setSel(String(r.run_id))}>{String(r.run_id)} <small>{String(r.status)} · {String(r.steps ?? '-')} steps · {String(r.gpu_model ?? '-')}</small></li>
          ))}
          {d.runs.length === 0 && <li className="muted">Generate a config first.</li>}
        </ul>
      </div>
      <div>
        {sel && run && (
          <>
            <div className="row">
              <button className="primary" onClick={() => void launch(false)}>Train (ai-toolkit on the GPU box)</button>
              <button onClick={() => void launch(true)}>Simulate (mock)</button>
              <button onClick={() => void writeNote()}>Write run note</button>
              <input placeholder="chosen checkpoint (e.g. 1500)" value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} />
              <button onClick={() => void pick()}>Save pick</button>
            </div>
            <div className="muted">status: {String(run.run.status)} {run.progress ? `· ${JSON.stringify(run.progress)}` : ''}</div>
            <pre className="log">{run.log || '(no log yet)'}</pre>
            {run.report && <details open><summary>Run note</summary><pre>{run.report}</pre></details>}
          </>
        )}
      </div>
    </div>
  )
}
