/* Highlight Reels app shell: the left rail, the stage stepper (Review > Fix-ups > Build > Publish),
   the background-jobs indicator and the Ctrl K jump palette. Shared by every page:
   <script src="/static/shell.js" data-page="review"></script> at the end of <body>.
   A page with a stepper puts <ol class="hr-steps" id="hrSteps"></ol> in its top bar.
   Pages talk to it through window.hrShell (setActive, refreshStages). */
(function () {
  const me = document.currentScript;
  const page = (me && me.dataset.page) || '';
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const REACH = 1.5;

  // Resolution badge for a video: SD / HD / FHD / QHD / 4K by its short side, exact size in the tooltip.
  // What a play is called: the name typed for it on the review page (review.notes), else its file name.
  window.hrClipName = (c, s) => String((s || c.review || {}).notes || '').trim().replace(/\s+/g, ' ') || c.source.split('/').pop();
  window.hrRes = (w, h, fps) => {
    const short = Math.min(w || 0, h || 0);
    if (!short) return '';
    const [lbl, cls] = short >= 2160 ? ['4K', 'uhd'] : short >= 1440 ? ['QHD', 'qhd'] : short >= 1080 ? ['FHD', 'fhd'] : short >= 720 ? ['HD', 'hd'] : ['SD', 'sd'];
    const tip = `${w}×${h}${fps ? ' · ' + Math.round(fps) + ' fps' : ''}${h > w ? ' · vertical' : ''}`;
    return `<span class="hr-res ${cls}" title="${tip}" aria-label="${lbl}, ${tip}">${lbl}</span>`;
  };   // a spotlight click this close to the snap freezes there (segment.FREEZE_REACH), as on /mark

  const ICONS = {
    projects: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    review: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9.5v5l4-2.5z"/>',
    fixups: '<path d="M12 3l9.5 17h-19z"/><path d="M12 10v4M12 17.5h.01"/>',
    build: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    publish: '<path d="M12 15V3"/><path d="M7 8l5-5 5 5"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
    players: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
    guide: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6"/><path d="M12 17.5h.01"/>',
    jobs: '<path d="M21 12a9 9 0 1 1-6.2-8.56"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
    check: '<path d="M5 12l5 5 9-10"/>',
  };
  const icon = (k, sw) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${sw || 1.8}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[k]}</svg>`;

  // ---------- rail ----------
  const NAV = [
    { k: 'projects', label: 'Projects', href: '/projects' },
    { k: 'review', label: 'Review', href: '/review', project: true },
    { k: 'fixups', label: 'Fix-ups', href: '/mark', project: true },
    { k: 'build', label: 'Build reel', href: '/review#build', project: true },
    { k: 'publish', label: 'Renders & publish', href: '/review#publish', project: true },
    'sep',
    { k: 'players', label: 'Player profiles', href: '/players' },
  ];
  const rail = document.createElement('nav');
  rail.className = 'hr-rail';
  rail.setAttribute('aria-label', 'App');
  rail.innerHTML =
    `<a class="hr-rail-logo" href="/projects" aria-label="Highlight Reels: projects"><svg width="22" height="19" viewBox="0 0 30 26" aria-hidden="true"><path d="M7 1h22l-7 24H0z" fill="currentColor"/><path d="M11.5 7.5l7.5 5.5-7.5 5.5z" fill="#0A0C0F"/></svg></a>` +
    NAV.map(n => n === 'sep' ? '<span class="hr-rail-sep"></span>' :
      `<a class="hr-rail-btn" href="${n.href}" data-k="${n.k}"${n.project ? ' data-project' : ''} aria-label="${n.label}" title="${n.label}">${icon(n.k)}<span class="hr-rail-badge" hidden></span></a>`).join('') +
    `<span class="hr-rail-fill"></span>
     <button type="button" class="hr-rail-btn" id="hrJobs" aria-label="Activity and log" title="Activity: nothing running">${icon('jobs')}<span class="hr-rail-dot" hidden></span></button>
     <button type="button" class="hr-rail-btn" id="hrFind" aria-label="Jump to (Ctrl K)" title="Jump to… (Ctrl K)">${icon('search')}</button>
     <a class="hr-rail-btn" href="/guide" data-k="guide" aria-label="Guide and shortcuts" title="Guide and shortcuts">${icon('guide')}</a>`;
  document.body.prepend(rail);
  document.documentElement.classList.add('has-rail');

  function setActive(k) {
    rail.querySelectorAll('[data-k]').forEach(a => {
      const on = a.dataset.k === k;
      a.classList.toggle('on', on);
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (steps) renderSteps(k);
  }

  // ---------- stage stepper ----------
  const steps = document.getElementById('hrSteps');
  let counts = null;
  const snapOf = c => c.review.snap ?? c.detection.snap;
  const fixupsOf = clips => clips.filter(c => c.review.keep !== false && (
    !(c.review.spotlight || []).some(a => Math.abs(a.t - snapOf(c)) <= REACH) ||
    !(c.review.hit != null || c.review.no_hit))).length;

  function renderSteps(active) {
    if (!steps) return;
    const c = counts || { total: 0, confirmed: 0, fixups: 0, inReel: 0 };
    const reviewed = c.total > 0 && c.confirmed >= c.total;
    const fixed = reviewed && c.fixups === 0;
    const S = [
      { k: 'review', label: 'Review', href: '/review', done: reviewed, meta: `<span class="hr-step-meta">${c.confirmed}/${c.total}</span>` },
      { k: 'fixups', label: 'Fix-ups', href: '/mark', done: fixed, meta: c.fixups ? `<span class="hr-step-badge">${c.fixups}</span>` : '' },
      { k: 'build', label: 'Build', href: '/review#build', done: false, meta: '' },
      { k: 'publish', label: 'Publish', href: '/review#publish', done: false, meta: '' },
    ];
    steps.innerHTML = S.map((s, i) => {
      const cur = s.k === active;
      const dot = s.done ? `<span class="hr-step-dot done">${icon('check', 3.5)}</span>` :
        `<span class="hr-step-dot${cur ? ' cur' : ''}${s.k === 'fixups' && c.fixups && reviewed ? ' warn' : ''}">${i + 1}</span>`;
      return (i ? '<li class="hr-step-sep" aria-hidden="true">›</li>' : '') +
        `<li><a class="hr-step${cur ? ' on' : ''}" href="${s.href}" data-k="${s.k}"${cur ? ' aria-current="step"' : ''}>${dot}${s.label}${s.meta}</a></li>`;
    }).join('');
  }

  async function refreshStages(clips) {
    try {
      if (!clips) {
        const r = await fetch('/api/clips');
        if (!r.ok) return;
        clips = (await r.json()).clips;
      }
      counts = {
        total: clips.length,
        confirmed: clips.filter(c => (c.review || {}).labeled).length,
        inReel: clips.filter(c => c.review.keep !== false).length,
        fixups: fixupsOf(clips),
      };
      renderSteps(current);
      const b = rail.querySelector('[data-k="fixups"] .hr-rail-badge');
      b.hidden = !counts.fixups; b.textContent = counts.fixups;
    } catch (e) { /* the stepper just shows zeros */ }
  }

  // ---------- background jobs ----------
  const jobsBtn = document.getElementById('hrJobs');
  let jobTimer = null;
  const jobLine = (what, j) => {
    const pct = j.total ? Math.round(100 * (j.done || 0) / j.total) : null;
    return what + (pct != null ? ` · ${pct}%` : '') + (j.step ? ` · ${j.step}` : '');
  };
  async function pollJobs() {
    clearTimeout(jobTimer);
    let running = false, text = 'Nothing running';
    try {
      const j = await (await fetch('/api/log?after=999999999')).json();
      if (j.setup && j.setup.state === 'running') { running = true; text = jobLine('Setting up a project', j.setup); }
      else if (j.reel && j.reel.state === 'running') { running = true; text = jobLine('Building the reel', j.reel); }
    } catch (e) {}
    jobsBtn.querySelector('.hr-rail-dot').hidden = !running;
    jobsBtn.classList.toggle('busy', running);
    jobsBtn.title = 'Activity: ' + text;
    jobsBtn.setAttribute('aria-label', 'Activity and log: ' + text);
    jobTimer = setTimeout(pollJobs, running ? 3000 : 15000);
  }

  // ---------- activity panel: what's running now, and the app's log ----------
  let act = null, actTimer = null, actLast = 0, actText = [];
  const ago = s => s < 90 ? Math.round(s) + ' s' : s < 5400 ? Math.round(s / 60) + ' min' : Math.round(s / 3600) + ' h';
  function jobCard(title, j, link) {
    if (!j || !j.state || j.state === 'idle') return '';
    const pct = j.total ? Math.round(100 * (j.done || 0) / j.total) : 0;
    const head = j.state === 'running' ? `${esc(j.step || 'working')} · ${j.done || 0} of ${j.total || 1}`
      : j.state === 'error' ? `Stopped: ${esc(j.error || 'unknown error')}` : 'Finished';
    return `<div class="hr-act-job ${j.state}"><b>${title}</b>${j.folder ? `<span class="hr-act-sub">${esc(j.folder)}</span>` : ''}
      <span>${head}</span>${j.state === 'running' ? `<div class="hr-act-bar"><div style="width:${pct}%"></div></div>` : ''}${link || ''}</div>`;
  }
  async function drawAct() {
    clearTimeout(actTimer);
    if (!act || act.hidden) return;
    let busy = false;
    try {
      const j = await (await fetch('/api/log?after=' + actLast)).json();
      busy = (j.setup && j.setup.state === 'running') || (j.reel && j.reel.state === 'running');
      const cards = jobCard('New project', j.setup) + jobCard('Reel build', j.reel, '<a href="/review#publish">Open renders</a>');
      const quiet = j.quiet == null ? '' : `Last log line ${ago(j.quiet)} ago.`;
      document.getElementById('hrActNow').innerHTML = (cards || '<div class="hr-act-job"><b>Nothing is running.</b></div>') +
        `<p class="hr-act-note">${busy ? 'Working. ' : ''}${quiet}${busy && j.quiet > 600 ? ' <b>No output for a long time: it may be stuck.</b>' : ''}</p>`;
      const box = document.getElementById('hrActLog');
      const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
      if (j.last < actLast) { box.textContent = ''; actText = []; }   // the app restarted: start the list again
      j.lines.forEach(l => {
        const d = document.createElement('div'), at = new Date(l.t * 1000).toLocaleTimeString();
        actText.push(at + '  ' + l.text);
        if (/FAIL|Traceback|Error|skipped/.test(l.text)) d.className = 'bad';
        d.innerHTML = `<time>${at}</time>${esc(l.text)}`;
        box.appendChild(d);
      });
      actLast = j.last;
      if (j.lines.length && atEnd) box.scrollTop = box.scrollHeight;
      document.getElementById('hrActFile').textContent = j.file ? 'Also saved in ' + j.file : '';
    } catch (e) {
      document.getElementById('hrActNow').innerHTML = '<div class="hr-act-job error"><b>The app isn\'t answering.</b><span>Start it again with the Highlight Reels shortcut.</span></div>';
    }
    actTimer = setTimeout(drawAct, busy ? 1000 : 4000);
  }
  function openAct() {
    if (!act) {
      act = document.createElement('aside');
      act.className = 'hr-act';
      act.setAttribute('aria-label', 'Activity and log');
      act.innerHTML = `<header><h2>Activity</h2><button type="button" class="hr-ghost" id="hrActCopy">Copy log</button>
        <button type="button" class="hr-ghost" id="hrActClose" aria-label="Close">Close</button></header>
        <div id="hrActNow"></div><div class="hr-act-log" id="hrActLog" role="log" aria-live="off"></div><footer id="hrActFile"></footer>`;
      document.body.appendChild(act);
      document.getElementById('hrActClose').onclick = () => { act.hidden = true; };
      document.getElementById('hrActCopy').onclick = e => {
        navigator.clipboard.writeText(actText.join('\n'));
        e.target.textContent = 'Copied'; setTimeout(() => { e.target.textContent = 'Copy log'; }, 1500);
      };
    }
    act.hidden = false;
    drawAct();
  }
  jobsBtn.onclick = () => { if (act && !act.hidden) act.hidden = true; else openAct(); };

  // ---------- Ctrl K palette ----------
  let pal = null, palItems = [], palSel = 0;
  function buildPalette() {
    pal = document.createElement('div');
    pal.className = 'hr-pal';
    pal.hidden = true;
    pal.innerHTML = `<div class="hr-pal-box" role="dialog" aria-modal="true" aria-label="Jump to">
      <label class="hr-pal-in">${icon('search', 2)}<input id="hrPalQ" type="text" placeholder="Jump to a play, project or page…" autocomplete="off" aria-label="Search"><kbd>Esc</kbd></label>
      <ul class="hr-pal-list" id="hrPalList" role="listbox"></ul></div>`;
    document.body.appendChild(pal);
    pal.addEventListener('mousedown', e => { if (e.target === pal) closePal(); });
    const q = document.getElementById('hrPalQ');
    q.addEventListener('input', () => { palSel = 0; drawPal(); });
    q.addEventListener('keydown', e => {
      const shown = filtered();
      if (e.key === 'ArrowDown') { e.preventDefault(); palSel = Math.min(palSel + 1, shown.length - 1); drawPal(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); palSel = Math.max(palSel - 1, 0); drawPal(); }
      else if (e.key === 'Enter') { e.preventDefault(); if (shown[palSel]) go(shown[palSel]); }
      else if (e.key === 'Escape') { e.preventDefault(); closePal(); }
      e.stopPropagation();
    });
    document.getElementById('hrPalList').addEventListener('click', e => {
      const li = e.target.closest('[data-i]');
      if (li) go(filtered()[+li.dataset.i]);
    });
  }
  const filtered = () => {
    const q = (document.getElementById('hrPalQ').value || '').trim().toLowerCase();
    const words = q.split(/\s+/).filter(Boolean);
    return palItems.filter(it => words.every(w => (it.label + ' ' + (it.sub || '') + ' ' + it.group).toLowerCase().includes(w))).slice(0, 40);
  };
  function drawPal() {
    const shown = filtered();
    let last = '';
    document.getElementById('hrPalList').innerHTML = shown.length ? shown.map((it, i) => {
      const head = it.group !== last ? `<li class="hr-pal-group" role="presentation">${esc(it.group)}</li>` : '';
      last = it.group;
      return head + `<li class="hr-pal-item${i === palSel ? ' on' : ''}" role="option" aria-selected="${i === palSel}" data-i="${i}">
        <span>${esc(it.label)}</span>${it.sub ? `<span class="hr-pal-sub">${esc(it.sub)}</span>` : ''}</li>`;
    }).join('') : '<li class="hr-pal-empty">Nothing matches.</li>';
    const on = document.querySelector('.hr-pal-item.on');
    if (on) on.scrollIntoView({ block: 'nearest' });
  }
  async function go(it) {
    closePal();
    if (it.act) { openAct(); return; }
    if (it.project) {
      if (it.project.current) { location = '/review'; return; }
      const r = await fetch('/api/projects/open', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ folder: it.project.folder }) });
      if (!r.ok) { alert((await r.json()).error || 'Could not open that project.'); return; }
      location = '/review';
      return;
    }
    if (it.href) {
      const here = location.pathname === it.href.split('#')[0] || (location.pathname === '/' && it.href.startsWith('/review'));
      location = it.href;
      if (here && it.href.includes('#')) window.dispatchEvent(new HashChangeEvent('hashchange'));
    }
  }
  async function openPal() {
    if (!pal) buildPalette();
    palItems = [
      { group: 'Go to', label: 'Projects', href: '/projects' },
      { group: 'Go to', label: 'Review plays', href: '/review' },
      { group: 'Go to', label: 'Fix-ups', sub: 'click your player, pick the hit', href: '/mark' },
      { group: 'Go to', label: 'Build reel', href: '/review#build' },
      { group: 'Go to', label: 'Renders & publish', sub: 'videos, YouTube', href: '/review#publish' },
      { group: 'Go to', label: 'Player profiles', href: '/players' },
      { group: 'Go to', label: 'Guide and shortcuts', href: '/guide' },
      { group: 'Go to', label: 'Activity and log', sub: 'what the app is doing', act: true },
    ];
    pal.hidden = false;
    const q = document.getElementById('hrPalQ');
    q.value = ''; palSel = 0; drawPal(); q.focus();
    try {
      const [cl, pr] = await Promise.all([
        fetch('/api/clips').then(r => r.ok ? r.json() : null).catch(() => null),
        fetch('/api/projects').then(r => r.ok ? r.json() : null).catch(() => null),
      ]);
      if (cl && cl.clips) cl.clips.forEach(c => palItems.push({
        group: 'Plays', label: hrClipName(c), sub: [c.group, c.review.keep === false ? 'left out' : ''].filter(Boolean).join(' · '),
        href: '/review#clip=' + encodeURIComponent(c.id) }));
      if (pr && pr.projects) pr.projects.filter(p => p.ready).forEach(p => palItems.push({
        group: 'Projects', label: (p.player && p.player.name) || p.name, sub: p.name + (p.current ? ' · open now' : ''), project: p }));
      if (!pal.hidden) drawPal();
    } catch (e) {}
  }
  function closePal() { if (pal) pal.hidden = true; }
  document.getElementById('hrFind').onclick = openPal;
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); if (pal && !pal.hidden) closePal(); else openPal(); }
  }, true);

  // ---------- start ----------
  let current = page;
  window.hrShell = {
    setActive(k) { current = k; setActive(k); },
    refreshStages,
    pollJobs,
    openActivity: openAct,
  };
  setActive(page);
  if (steps) refreshStages();
  else fetch('/api/clips').then(r => r.ok ? r.json() : null).then(j => {   // just the Fix-ups badge on the rail
    if (!j || !j.clips) { rail.querySelectorAll('[data-project]').forEach(a => a.classList.add('dim')); return; }
    const n = fixupsOf(j.clips), b = rail.querySelector('[data-k="fixups"] .hr-rail-badge');
    b.hidden = !n; b.textContent = n;
  }).catch(() => {});
  pollJobs();
})();
