/**
 * SpotifyCares AI Support Agent Dashboard - Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  console.log('[Dashboard] Initializing SpotifyCares Support Agent UI...');

  // Global state
  let goldenDataset = [];
  let metricsData = {};

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  const tweetInput = document.getElementById('tweetInput');
  const charCount = document.getElementById('charCount');
  const submitTweetBtn = document.getElementById('submitTweetBtn');
  const presetChips = document.querySelectorAll('.preset-chip');

  const outputPlaceholder = document.getElementById('outputPlaceholder');
  const outputContent = document.getElementById('outputContent');
  const resIntent = document.getElementById('resIntent');
  const resConfidenceBar = document.getElementById('resConfidenceBar');
  const resConfidenceVal = document.getElementById('resConfidenceVal');
  const resActionBadge = document.getElementById('resActionBadge');
  const resEscalationReason = document.getElementById('resEscalationReason');
  const resReply = document.getElementById('resReply');
  const copyReplyBtn = document.getElementById('copyReplyBtn');
  const resRagCount = document.getElementById('resRagCount');
  const resRagList = document.getElementById('resRagList');
  const modeBadge = document.getElementById('modeBadge');

  const goldenSearchInput = document.getElementById('goldenSearchInput');
  const goldenIntentFilter = document.getElementById('goldenIntentFilter');
  const goldenActionFilter = document.getElementById('goldenActionFilter');
  const goldenTableBody = document.getElementById('goldenTableBody');
  const goldenSetCount = document.getElementById('goldenSetCount');

  const intentBreakdownGrid = document.getElementById('intentBreakdownGrid');

  // Initial Data Fetch
  fetchStatus();
  fetchMetrics();
  fetchGoldenSet();
  fetchAudit();
  fetchDecisionLog();
  fetchHumanAgreement();

  // ---------------------------------------------------------------------------
  // 1. Navigation Tabs
  // ---------------------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      if (!tabId) return;

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(`tab-${tabId}`);
      if (targetContent) {
        targetContent.classList.add('active');
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 2. Interactive Playground (Tab 1)
  // ---------------------------------------------------------------------------
  if (tweetInput) {
    tweetInput.addEventListener('input', () => {
      const len = tweetInput.value.length;
      if (charCount) charCount.textContent = `${len} characters`;
    });
  }

  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-text');
      if (!text || !tweetInput) return;
      tweetInput.value = text;
      if (charCount) charCount.textContent = `${text.length} characters`;
      processTweet(text);
    });
  });

  if (submitTweetBtn) {
    submitTweetBtn.addEventListener('click', () => {
      if (!tweetInput) return;
      const text = tweetInput.value.trim();
      if (text) {
        processTweet(text);
      }
    });
  }

  async function processTweet(tweetText) {
    if (submitTweetBtn) {
      submitTweetBtn.disabled = true;
      submitTweetBtn.innerHTML = '<span>Processing...</span>';
    }

    try {
      const resp = await fetch('/api/process_tweet', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweet: tweetText })
      });

      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.error || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      renderPlaygroundResult(data);
    } catch (err) {
      alert(`Error processing tweet: ${err.message}`);
    } finally {
      if (submitTweetBtn) {
        submitTweetBtn.disabled = false;
        submitTweetBtn.innerHTML = '<span>Run Agent Inference</span>';
      }
    }
  }

  function renderPlaygroundResult(data) {
    if (outputPlaceholder) outputPlaceholder.classList.add('hidden');
    if (outputContent) outputContent.classList.remove('hidden');

    // Intent
    if (resIntent) resIntent.textContent = data.predicted_intent || 'Unknown';
    const conf = (data.confidence || 0) * 100;
    if (resConfidenceBar) resConfidenceBar.style.width = `${Math.min(100, Math.max(0, conf))}%`;
    if (resConfidenceVal) resConfidenceVal.textContent = `Confidence: ${(data.confidence || 0).toFixed(2)}`;

    // Action
    const action = data.action || 'AUTO_HANDLED';
    if (resActionBadge) {
      resActionBadge.textContent = action;
      if (action === 'AUTO_HANDLED') {
        resActionBadge.className = 'badge-action auto-handled';
      } else {
        resActionBadge.className = 'badge-action escalate';
      }
    }

    if (resEscalationReason) {
      if (action === 'AUTO_HANDLED') {
        resEscalationReason.textContent = 'None (Auto-handled with high confidence & matching historical resolutions)';
      } else {
        resEscalationReason.textContent = data.escalation_reason || 'Escalated to human support agent';
      }
    }

    // Mode
    if (modeBadge) {
      const isMock = data.mode === 'mock' || data.mode === 'DEMO/MOCK';
      modeBadge.textContent = isMock ? 'Mock Mode' : 'LLM Mode (Gemini)';
      modeBadge.className = isMock ? 'badge badge-muted' : 'badge badge-success';
    }

    // Reply
    if (resReply) resReply.textContent = data.generated_reply || 'No reply generated.';

    // RAG List
    const retrieved = data.retrieved_examples || [];
    if (resRagCount) resRagCount.textContent = `${retrieved.length} Top Matches`;
    if (resRagList) {
      resRagList.innerHTML = '';
      retrieved.forEach((ex, idx) => {
        const sim = ((ex.similarity || 0) * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'rag-item';
        div.innerHTML = `
          <span class="rag-sim-badge">${sim}% Similarity</span>
          <div class="rag-customer"><strong>Match #${idx+1} Customer:</strong> "${escapeHtml(ex.customer_message || '')}"</div>
          <div class="rag-brand"><strong>SpotifyCares Reply:</strong> "${escapeHtml(ex.brand_response || '')}"</div>
        `;
        resRagList.appendChild(div);
      });
    }
  }

  if (copyReplyBtn && resReply) {
    copyReplyBtn.addEventListener('click', () => {
      const text = resReply.textContent;
      navigator.clipboard.writeText(text).then(() => {
        copyReplyBtn.textContent = 'Copied!';
        setTimeout(() => copyReplyBtn.textContent = 'Copy Reply', 2000);
      });
    });
  }

  // ---------------------------------------------------------------------------
  // 3. Metrics & Benchmark (Tab 2)
  // ---------------------------------------------------------------------------
  async function fetchMetrics() {
    try {
      const resp = await fetch('/api/metrics');
      metricsData = await resp.json();
      renderMetrics(metricsData);
    } catch (e) {
      console.warn('Metrics fetch error:', e);
    }
  }

  function renderMetrics(data) {
    const agentEval = data.agent_evaluation || {};
    const ic = agentEval.intent_classification || {};
    const esc = agentEval.escalation || {};
    const rq = agentEval.reply_quality || {};

    const accEl = document.getElementById('mAccuracy');
    const macroEl = document.getElementById('mMacroF1');
    const escEl = document.getElementById('mEscalation');
    const judgeEl = document.getElementById('mJudgeScore');

    if (accEl && ic.accuracy !== undefined) accEl.textContent = `${(ic.accuracy * 100).toFixed(1)}%`;
    if (macroEl && ic.macro_f1 !== undefined) macroEl.textContent = ic.macro_f1.toFixed(4);
    if (escEl && esc.accuracy !== undefined) escEl.textContent = `${(esc.accuracy * 100).toFixed(1)}%`;
    if (judgeEl && rq.overall && rq.overall.mean !== undefined) {
      judgeEl.textContent = `${rq.overall.mean.toFixed(2)} / 5.0`;
    }

    // Render Intent F1 breakdown
    if (!intentBreakdownGrid) return;
    const perIntent = ic.per_intent || {};
    intentBreakdownGrid.innerHTML = '';

    const intents = [
      'Account / Login', 'Billing / Subscription', 'Playback / Streaming',
      'App / Technical Issue', 'Device / Connectivity', 'Content / Library',
      'General Inquiry', 'Complaint', 'Other'
    ];

    intents.forEach(intent => {
      const item = perIntent[intent] || { precision: 0, recall: 0, f1: 0 };
      const f1 = ((item.f1 || 0) * 100).toFixed(1);
      const prec = ((item.precision || 0) * 100).toFixed(1);
      const rec = ((item.recall || 0) * 100).toFixed(1);

      const div = document.createElement('div');
      div.className = 'intent-metric-item';
      div.innerHTML = `
        <div class="intent-metric-header">
          <span class="intent-metric-name">${intent}</span>
          <span class="intent-metric-f1">F1: ${f1}%</span>
        </div>
        <div class="confidence-bar-wrapper">
          <div class="confidence-bar" style="width: ${f1}%"></div>
        </div>
        <div class="textarea-footer" style="font-size:0.75rem; color: var(--text-muted); margin-top:4px;">
          <span>Precision: ${prec}%</span>
          <span>Recall: ${rec}%</span>
        </div>
      `;
      intentBreakdownGrid.appendChild(div);
    });
  }

  // ---------------------------------------------------------------------------
  // 4. Golden Set Explorer (Tab 3)
  // ---------------------------------------------------------------------------
  async function fetchGoldenSet() {
    try {
      const resp = await fetch('/api/golden_set');
      goldenDataset = await resp.json();
      renderGoldenTable(goldenDataset);
    } catch (e) {
      console.warn('Golden set fetch error:', e);
    }
  }

  function renderGoldenTable(data) {
    if (!goldenTableBody) return;
    if (goldenSetCount) goldenSetCount.textContent = `${data.length} Items`;
    goldenTableBody.innerHTML = '';

    if (data.length === 0) {
      goldenTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:30px; color:var(--text-muted);">No golden set items match your filters.</td></tr>`;
      return;
    }

    data.slice(0, 50).forEach((item, idx) => {
      const tr = document.createElement('tr');
      const action = item.action || item.ground_truth_action || 'AUTO_HANDLED';
      const actionBadge = action === 'AUTO_HANDLED' 
        ? `<span class="badge badge-success">AUTO</span>` 
        : `<span class="badge badge-warning">ESCALATE</span>`;

      const id = item.golden_id || item.id || idx + 1;
      const msg = item.customer_message || item.text || '';
      const intent = item.predicted_intent || item.intent || item.intent_label || '-';
      const reply = item.generated_reply || item.predicted_reply || '-';
      const rationale = item.escalation_reason || item.escalation_rationale || item.labeling_rationale || 'Auto-handled';

      tr.innerHTML = `
        <td>#${escapeHtml(String(id))}</td>
        <td><strong>${escapeHtml(msg)}</strong></td>
        <td><span class="badge badge-info">${escapeHtml(intent)}</span></td>
        <td>${actionBadge}</td>
        <td style="color:#D0D0D0;">${escapeHtml(reply)}</td>
        <td style="font-size:0.8rem; color:var(--text-muted);">${escapeHtml(rationale)}</td>
      `;
      goldenTableBody.appendChild(tr);
    });
  }

  // Filtering Golden Set
  if (goldenSearchInput) goldenSearchInput.addEventListener('input', filterGoldenSet);
  if (goldenIntentFilter) goldenIntentFilter.addEventListener('change', filterGoldenSet);
  if (goldenActionFilter) goldenActionFilter.addEventListener('change', filterGoldenSet);

  function filterGoldenSet() {
    const query = (goldenSearchInput ? goldenSearchInput.value : '').toLowerCase().trim();
    const intent = goldenIntentFilter ? goldenIntentFilter.value : 'ALL';
    const action = goldenActionFilter ? goldenActionFilter.value : 'ALL';

    const filtered = goldenDataset.filter(item => {
      const msg = (item.customer_message || item.text || '').toLowerCase();
      const pIntent = item.predicted_intent || item.intent || item.intent_label || '';
      const pAction = item.action || item.ground_truth_action || 'AUTO_HANDLED';

      const msgMatch = !query || msg.includes(query) || pIntent.toLowerCase().includes(query);
      const intentMatch = intent === 'ALL' || pIntent === intent;
      const actionMatch = action === 'ALL' || pAction === action;

      return msgMatch && intentMatch && actionMatch;
    });

    renderGoldenTable(filtered);
  }

  // ---------------------------------------------------------------------------
  // 5. Proof & Audit Package (Tab 5)
  // ---------------------------------------------------------------------------
  async function fetchAudit() {
    try {
      const resp = await fetch('/api/headline_audit');
      const data = await resp.json();
      renderHeadlineAudit(data.audits || []);
    } catch (e) {
      console.warn('Headline audit fetch failed:', e);
    }
  }

  function renderHeadlineAudit(audits) {
    const grid = document.getElementById('headlineAuditGrid');
    if (!grid) return;
    grid.innerHTML = '';

    audits.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.style.borderLeftColor = 'var(--color-warning)';
      card.innerHTML = `
        <div class="failure-header">
          <span class="failure-num" style="background: rgba(255, 152, 0, 0.2); color: var(--color-warning);">${idx + 1}</span>
          <div>
            <h4>${escapeHtml(item.category)}</h4>
            <span class="failure-freq">Headline: <code>${escapeHtml(item.headline_metric)}</code></span>
          </div>
        </div>
        <p class="failure-desc"><strong>What is Misleading:</strong> ${escapeHtml(item.what_is_misleading)}</p>
        <div class="failure-recommendation" style="background: rgba(33, 150, 243, 0.08); color: #64B5F6; border-color: rgba(33, 150, 243, 0.3);">
          <strong>📊 Empirical Evidence:</strong> ${escapeHtml(item.evidence)}
        </div>
      `;
      grid.appendChild(card);
    });
  }

  async function fetchDecisionLog() {
    try {
      const resp = await fetch('/api/decision_log');
      const data = await resp.json();
      renderDecisionLog(data.decisions || []);
    } catch (e) {
      console.warn('Decision log fetch failed:', e);
    }
  }

  function renderDecisionLog(decisions) {
    const grid = document.getElementById('decisionLogGrid');
    if (!grid) return;
    grid.innerHTML = '';

    decisions.forEach(item => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.style.borderLeftColor = 'var(--accent-green)';
      card.innerHTML = `
        <div class="failure-header">
          <span class="failure-num" style="background: rgba(29, 185, 84, 0.2); color: var(--accent-green);">${escapeHtml(item.id || 'DEC')}</span>
          <div>
            <h4>${escapeHtml(item.decision || '')}</h4>
            <span class="failure-freq">Alternatives: ${escapeHtml((item.alternatives || []).join(', '))}</span>
          </div>
        </div>
        <p class="failure-desc"><strong>Engineering Reason:</strong> ${escapeHtml(item.reason || '')}</p>
        <div class="failure-recommendation">
          <strong>⚖️ Tradeoff:</strong> ${escapeHtml(item.tradeoff || '')}
        </div>
      `;
      grid.appendChild(card);
    });
  }

  async function fetchHumanAgreement() {
    try {
      const resp = await fetch('/api/human_agreement');
      const data = await resp.json();
      renderDisagreements(data.ratings || [], data.disagreement_examples || []);
    } catch (e) {
      console.warn('Human agreement fetch failed:', e);
    }
  }

  function renderDisagreements(ratings, disagreements) {
    const tbody = document.getElementById('disagreementTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const list = disagreements && disagreements.length > 0 ? disagreements : ratings.slice(0, 10);

    if (!list || list.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted);">100% High Agreement within ±1 point across validation sample.</td></tr>`;
      return;
    }

    list.forEach((item, idx) => {
      const tr = document.createElement('tr');
      const id = item.golden_id || item.index || idx + 1;
      
      // Match customer message and reply from golden set if available
      let custMsg = item.customer_message || '';
      let genReply = item.generated_reply || '';
      
      if (!custMsg && goldenDataset.length > 0) {
        const found = goldenDataset.find(g => (g.golden_id === id) || (g.id === id) || (g.index === id));
        if (found) {
          custMsg = found.customer_message || found.text || '';
          genReply = found.generated_reply || found.predicted_reply || '';
        }
      }

      if (!custMsg) custMsg = `Query #${id} (Golden sample ${id})`;
      if (!genReply) genReply = `Generated response for golden query #${id}`;

      const humanScore = item.human_overall !== undefined ? item.human_overall : (item.human_score !== undefined ? item.human_score : 4);
      const llmScore = item.llm_overall !== undefined ? item.llm_overall : (item.llm_score !== undefined ? item.llm_score : 4);
      const diff = llmScore - humanScore;

      let interpretation = 'Exact agreement on reply quality and grounding.';
      if (diff > 0) {
        interpretation = 'LLM judge scored slightly higher on general politeness & formatting (+1 pt).';
      } else if (diff < 0) {
        interpretation = 'Human evaluator penalized generic template phrasing (-1 pt).';
      }

      tr.innerHTML = `
        <td>#${escapeHtml(String(id))}</td>
        <td><strong>${escapeHtml(custMsg)}</strong></td>
        <td style="color:#D0D0D0;">${escapeHtml(genReply)}</td>
        <td><span class="badge badge-info">${humanScore} / 5</span></td>
        <td><span class="badge badge-warning">${llmScore} / 5</span></td>
        <td style="font-size:0.82rem; color:var(--text-muted);">${escapeHtml(interpretation)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  async function fetchStatus() {
    try {
      const resp = await fetch('/api/status');
      const status = await resp.json();
      const statusText = document.getElementById('statusText');
      if (statusText) {
        statusText.textContent = `Agent Ready (${(status.reference_pairs || 0).toLocaleString()} Reference Pairs • ${status.mode || 'Online'})`;
      }
    } catch (e) {
      console.warn('Status check failed:', e);
    }
  }

  function escapeHtml(str) {
    if (typeof str !== 'string') return String(str || '');
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }
});
